import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import * as Tone from 'tone';
import { generateImageWithPrompt, generateImageFromText, getRemixSuggestions } from './services/geminiService';

// --- SOUND DEFINITIONS ---
const synth = new Tone.Synth({
  oscillator: { type: 'sine' },
  envelope: {
    attack: 0.01,
    decay: 0.2,
    sustain: 0,
    release: 0.2,
  },
}).toDestination();


const ensureAudioContext = async () => {
    if (Tone.context.state !== 'running') {
        await Tone.start();
    }
};


// --- PROMPT DEFINITIONS ---
const PROMPT_PREFIX_IMAGE = "Concisely name the key entity in this image (e.g. person, object, building). ";
const PROMPT_POSTFIX = "in isometric perspective, 8-bit sprite on a white background. No drop shadow";
const PROMPT_MAIN = (subject: string) => `Create 3d pixel art of ${subject} `;
const IMAGE_PROMPT = PROMPT_PREFIX_IMAGE + PROMPT_MAIN("the isolated key entity") + PROMPT_POSTFIX;
const TEXT_PROMPT_TEMPLATE = (input: string) => PROMPT_MAIN(input) + PROMPT_POSTFIX;
const REMIX_PROMPT_TEMPLATE = (input: string) => `${input}. Keep it as 3d pixel art in isometric perspective, 8-bit sprite on white background. No drop shadow.`;
const REMIX_SUGGESTION_PROMPT = `Here is some 3d pixel art. Come up with 5 brief prompts for ways to remix the key entity/object. e.g. "Make it [x]" or "Add a [x]" or some other alteration of the key entity/object. Do NOT suggest ways to alter the environment or background, that must stay a plain solid empty background. Only give alterations of the key entity/object itself. Prompts should be under 8 words.`;


const IMAGE_WIDTH = 375; // Increased from 250

// Adjust this value to control how aggressively the background is removed.
// Higher values are more aggressive. Good for chroma key.
const COLOR_DISTANCE_THRESHOLD = 20;
const MOVE_AMOUNT = 25; // Corresponds to the vertical step of the isometric grid
const MIN_WIDTH = 50;
const MAX_WIDTH = 1000;
const MIN_SCALE = 0.2;
const MAX_SCALE = 3.0;

interface ProcessedImage {
  id: number;
  sourceFile?: File;
  sourceText?: string;
  processedImage: HTMLImageElement | null;
  originalImageUrl?: string;
  showOriginal?: boolean;
  x: number; // World coordinates
  y: number; // World coordinates
  width: number;
  height: number;
  isGenerating: boolean;
  contentBounds: { x: number; y: number; width: number; height: number; };
  sourcePreviewUrl?: string;
  flippedHorizontally?: boolean;
  isVariation?: boolean;
  remixSuggestions?: string[];
  generatingPrompt?: string; // To display during loading
}

interface ImageProcessingResult {
    transparentImage: HTMLImageElement;
    contentBounds: { x: number; y: number; width: number; height: number; };
}

const processImageForTransparency = (imageUrl: string): Promise<ImageProcessingResult> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        return reject(new Error('Could not get 2d context'));
      }
      ctx.drawImage(img, 0, 0);
      try {
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // Get top-left pixel color as the background color
        const bgR = data[0];
        const bgG = data[1];
        const bgB = data[2];

        // Using squared Euclidean distance for performance.
        const colorDistanceThresholdSquared = COLOR_DISTANCE_THRESHOLD * COLOR_DISTANCE_THRESHOLD;

        let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;

        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          
          const distanceSquared = (r - bgR) ** 2 + (g - bgG) ** 2 + (b - bgB) ** 2;

          if (distanceSquared < colorDistanceThresholdSquared) {
            data[i + 3] = 0; // Set alpha to 0 (transparent)
          }
        }
        
        // Calculate tight bounding box
        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                const alpha = data[(y * canvas.width + x) * 4 + 3];
                if (alpha > 0) { // If pixel is not transparent
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        ctx.putImageData(imageData, 0, 0);
        
        const transparentImage = new Image();
        transparentImage.src = canvas.toDataURL();
        transparentImage.onload = () => {
            const contentBounds = (maxX >= minX && maxY >= minY) 
                ? { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 }
                : { x: 0, y: 0, width: canvas.width, height: canvas.height }; // Fallback
            resolve({ transparentImage, contentBounds });
        };
        transparentImage.onerror = (err) => reject(err);

      } catch (error) {
         console.error("Error processing image for transparency:", error);
         // Resolve with original image if processing fails
         resolve({ transparentImage: img, contentBounds: { x: 0, y: 0, width: img.width, height: img.height }});
      }
    };
    img.onerror = (err) => reject(err);
    img.src = imageUrl;
  });
};

const drawIsometricGrid = (ctx: CanvasRenderingContext2D, width: number, height: number, scale: number) => {
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.5 / scale; // Keep line width consistent when zooming
    
    const isoAngle = Math.PI / 6; // 30 degrees
    const gridSpacing = 50;
    
    const sinAngle = Math.sin(isoAngle);
    const cosAngle = Math.cos(isoAngle);
    
    // Using a large number based on the diagonal to ensure coverage, adjusted for scale
    const extendedDim = (width + height) / scale * 1.5;

    ctx.beginPath();
    
    // Lines from top-left to bottom-right
    for (let i = -extendedDim; i < extendedDim; i += gridSpacing) {
        ctx.moveTo(i - extendedDim * cosAngle, 0 - extendedDim * sinAngle);
        ctx.lineTo(i + extendedDim * cosAngle, 0 + extendedDim * sinAngle);
    }

    // Lines from top-right to bottom-left
    for (let i = -extendedDim; i < extendedDim; i += gridSpacing) {
        ctx.moveTo(i + extendedDim * cosAngle, 0 - extendedDim * sinAngle);
        ctx.lineTo(i - extendedDim * cosAngle, 0 + extendedDim * sinAngle);
    }
    
    ctx.stroke();
};

const imageElementToFile = async (imageElement: HTMLImageElement, fileName: string): Promise<File> => {
    const canvas = document.createElement('canvas');
    canvas.width = imageElement.naturalWidth;
    canvas.height = imageElement.naturalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error("Could not get 2d context for image conversion");
    ctx.drawImage(imageElement, 0, 0);
    return new Promise((resolve, reject) => {
        canvas.toBlob(blob => {
            if (blob) {
                resolve(new File([blob], fileName, { type: 'image/png' }));
            } else {
                reject(new Error("Canvas to Blob conversion failed"));
            }
        }, 'image/png');
    });
};


const App: React.FC = () => {
  const [images, setImages] = useState<ProcessedImage[]>([]);
  const [textInput, setTextInput] = useState('');
  const [remixInput, setRemixInput] = useState('');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const remixInputRef = useRef<HTMLInputElement>(null);
  const [draggingImage, setDraggingImage] = useState<{ id: number; offsetX: number; offsetY: number } | null>(null);
  const [hoveredImageId, setHoveredImageId] = useState<number | null>(null);
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);
  const nextId = useRef(0);
  const previewImageCache = useRef<Record<number, HTMLImageElement>>({});
  const originalImageCache = useRef<Record<number, HTMLImageElement>>({});
  const prevImagesRef = useRef<ProcessedImage[]>([]);
  const [animationTick, setAnimationTick] = useState(0);
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [suggestionIndex, setSuggestionIndex] = useState(0);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [viewTransform, setViewTransform] = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [pinchState, setPinchState] = useState<{ 
    id: number | 'canvas'; 
    initialDistance: number; 
    initialWidth?: number; 
    initialHeight?: number; 
    centerX?: number; 
    centerY?: number;
    initialViewTransform?: { scale: number; offsetX: number; offsetY: number; };
    initialMidPoint?: { x: number; y: number; };
  } | null>(null);


  const selectedImage = useMemo(() =>
    selectedImageId !== null ? images.find(img => img.id === selectedImageId) : null,
    [images, selectedImageId]
  );

  const isActionDisabled = !selectedImage || selectedImage.isGenerating;

  // Effect for render complete sound
  useEffect(() => {
    const prevImagesMap = new Map(prevImagesRef.current.map(img => [img.id, img]));

    images.forEach(img => {
        const prevImg = prevImagesMap.get(img.id);
        if (prevImg && prevImg.isGenerating && !img.isGenerating) {
            // Image just finished generating
            (async () => {
                await ensureAudioContext();
                synth.triggerAttackRelease('C5', '8n');
            })();
        }
    });
  }, [images]);

  // Effect for smooth loader animations
  useEffect(() => {
    const isGenerating = images.some(img => img.isGenerating);
    let intervalId: number | undefined;
    if (isGenerating) {
        intervalId = window.setInterval(() => {
            setAnimationTick(tick => tick + 1);
        }, 200); // Faster update interval
    }
    return () => {
        if (intervalId) {
            clearInterval(intervalId);
        }
    };
  }, [images]);

  // Effect for cycling through remix suggestions
  useEffect(() => {
    if (selectedImage?.remixSuggestions?.length) {
        const intervalId = setInterval(() => {
            setSuggestionIndex(prev => (prev + 1) % (selectedImage.remixSuggestions?.length || 1));
        }, 3000); // 3-second timer
        return () => clearInterval(intervalId);
    }
  }, [selectedImage]);


  useEffect(() => {
    const prevUrls = new Set(prevImagesRef.current.map(i => i.sourcePreviewUrl).filter(Boolean));
    const currentUrls = new Set(images.map(i => i.sourcePreviewUrl).filter(Boolean));

    prevUrls.forEach(url => {
      // only revoke blob URLs, not data URLs
      if (url && url.startsWith('blob:') && !currentUrls.has(url)) {
        URL.revokeObjectURL(url);
        const entry = Object.entries(previewImageCache.current).find(([, img]) => img.src === url);
        if (entry) {
          delete previewImageCache.current[parseInt(entry[0], 10)];
        }
      }
    });
    
    // Cleanup for original image cache
    const prevIds = new Set(prevImagesRef.current.map(i => i.id));
    const currentIds = new Set(images.map(i => i.id));
    prevIds.forEach(id => {
      if (!currentIds.has(id)) {
        delete originalImageCache.current[id];
      }
    });

    prevImagesRef.current = images;
  }, [images]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.save();
    ctx.translate(viewTransform.offsetX, viewTransform.offsetY);
    ctx.scale(viewTransform.scale, viewTransform.scale);
    
    drawIsometricGrid(ctx, canvas.width, canvas.height, viewTransform.scale);

    const ellipses = ['.', '..', '...'][animationTick % 3];
    
    const drawLoaderEllipses = (x: number, y: number, size: 'large' | 'small' = 'large') => {
        ctx.font = size === 'large' ? `${36 / viewTransform.scale}px "Space Mono"` : `${24 / viewTransform.scale}px "Space Mono"`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3 / viewTransform.scale;
        ctx.strokeText(ellipses, x, y);
        ctx.fillStyle = 'white';
        ctx.fillText(ellipses, x, y);
    }

    images.forEach(img => {
      const drawX = Math.round(img.x);
      const drawY = Math.round(img.y);
      
      const imageToDraw = (img.showOriginal && originalImageCache.current[img.id])
          ? originalImageCache.current[img.id]
          : img.processedImage;

      if (imageToDraw && !img.isGenerating) {
        if (img.flippedHorizontally) {
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(imageToDraw, -drawX - img.width, drawY, img.width, img.height);
            ctx.restore();
        } else {
            ctx.drawImage(imageToDraw, drawX, drawY, img.width, img.height);
        }
      }
      
      if (img.isGenerating) {
        if (imageToDraw) { // --- REGENERATING / REMIXING ---
            if (img.flippedHorizontally) {
                ctx.save();
                ctx.scale(-1, 1);
                ctx.drawImage(imageToDraw, -drawX - img.width, drawY, img.width, img.height);
                ctx.restore();
            } else {
                ctx.drawImage(imageToDraw, drawX, drawY, img.width, img.height);
            }
            
            const loaderY = drawY + img.height + (20 / viewTransform.scale);

            if (img.isVariation && img.generatingPrompt) {
                 ctx.fillStyle = 'black';
                 ctx.font = `${14 / viewTransform.scale}px "Space Mono"`;
                 ctx.textAlign = 'center';
                 ctx.textBaseline = 'top';
                 ctx.fillText(img.generatingPrompt, drawX + img.width / 2, loaderY);
                 drawLoaderEllipses(drawX + img.width / 2, loaderY + (25 / viewTransform.scale));
            } else {
                drawLoaderEllipses(drawX + img.width / 2, drawY + img.height / 2);
            }

        } else if (img.sourcePreviewUrl && previewImageCache.current[img.id]) { // --- NEW IMAGE LOADER ---
            const previewImg = previewImageCache.current[img.id];
            if (previewImg.complete) {
                const PADDING = 10;
                const loaderWidth = 250 * 0.6; 
                const aspectRatio = previewImg.width / previewImg.height;
                const loaderHeight = loaderWidth / aspectRatio;
                const containerWidth = loaderWidth + PADDING * 2;
                const containerHeight = loaderHeight + PADDING * 2;
                const containerX = Math.round(img.x + (img.width - containerWidth) / 2);
                const containerY = Math.round(img.y + (img.height - containerHeight) / 2);
                
                ctx.drawImage(previewImg, containerX + PADDING, containerY + PADDING, loaderWidth, loaderHeight);
                
                drawLoaderEllipses(containerX + containerWidth / 2, containerY + containerHeight / 2, 'small');
            }
        } else if (img.sourceText) { // --- NEW TEXT LOADER ---
            const loaderWidth = 250; 
            const loaderHeight = 250;
            const loaderX = drawX;
            const loaderY = Math.round(img.y + (img.height - loaderHeight) / 2);
            
            const centerX = loaderX + loaderWidth / 2;
            const centerY = loaderY + loaderHeight / 2;
            const radius = loaderWidth / 2;

            ctx.strokeStyle = 'black';
            ctx.lineWidth = 1 / viewTransform.scale;
            ctx.setLineDash([4 / viewTransform.scale, 4 / viewTransform.scale]);
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.setLineDash([]);
            
            ctx.fillStyle = 'black';
            ctx.font = `${14 / viewTransform.scale}px "Space Mono"`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            const text = img.sourceText;
            const maxWidth = loaderWidth - 20;
            const words = text.split(' ');
            let line = '';
            const lines = [];
            
            for (let n = 0; n < words.length; n++) {
                const testLine = line + words[n] + ' ';
                const testWidth = ctx.measureText(testLine).width;
                if (testWidth > maxWidth && n > 0) {
                    lines.push(line);
                    line = words[n] + ' ';
                } else {
                    line = testLine;
                }
            }
            lines.push(line);
            
            const lineHeight = 18;
            const totalTextHeight = lines.length * lineHeight;
            let textY = loaderY + (loaderHeight - totalTextHeight) / 2 + (lineHeight / 2);
            
            for(const l of lines) {
                ctx.fillText(l.trim(), loaderX + loaderWidth / 2, Math.round(textY));
                textY += lineHeight;
            }

            const textBottomY = textY - lineHeight;
            const remainingSpace = (loaderY + loaderHeight) - textBottomY;
            const ellipsesY = textBottomY + remainingSpace / 2;

            drawLoaderEllipses(loaderX + loaderWidth / 2, Math.round(ellipsesY), 'small');
        }
      }
      
      const isHovered = hoveredImageId === img.id && !img.isGenerating && img.processedImage;
      if (selectedImageId === img.id || isHovered) {
        const centerX = Math.round(img.x + img.width / 2);
        const centerY = Math.round(img.y + img.height / 2);
        const radius = Math.min(img.width, img.height) / 2 * 0.8;
        
        ctx.lineWidth = 1 / viewTransform.scale;
        if (selectedImageId === img.id) {
            ctx.strokeStyle = 'rgba(0, 123, 255, 0.5)';
        } else { // isHovered
            ctx.strokeStyle = 'rgba(0, 123, 255, 0.25)';
        }

        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.stroke();
      }
    });

    ctx.restore();
  }, [images, hoveredImageId, selectedImageId, animationTick, viewTransform]);

  useEffect(() => {
      const animationFrameId = requestAnimationFrame(drawCanvas);
      return () => cancelAnimationFrame(animationFrameId);
  }, [drawCanvas, animationTick]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      drawCanvas();
    };
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    return () => window.removeEventListener('resize', resizeCanvas);
  }, [drawCanvas]);

  const generateFromImage = useCallback(async (file: File, id: number, prompt: string = IMAGE_PROMPT) => {
    try {
      const { imageUrl } = await generateImageWithPrompt(file, prompt);
      if (!imageUrl) throw new Error("Generation failed, no image returned.");

      const originalImg = new Image();
      originalImg.src = imageUrl;
      originalImageCache.current[id] = originalImg;
      
      const { transparentImage, contentBounds } = await processImageForTransparency(imageUrl);
      const aspectRatio = transparentImage.width / transparentImage.height;
      
      const imageFileForSuggestions = await imageElementToFile(transparentImage, 'suggestion-source.png');
      const suggestions = await getRemixSuggestions(imageFileForSuggestions, REMIX_SUGGESTION_PROMPT);

      setImages(prev => prev.map(img => {
        if (img.id !== id) return img;
        
        const newWidth = IMAGE_WIDTH;
        const newHeight = IMAGE_WIDTH / aspectRatio;
        const currentCenterX = img.x + img.width / 2;
        const currentCenterY = img.y + img.height / 2;

        return {
          ...img,
          processedImage: transparentImage,
          originalImageUrl: imageUrl,
          showOriginal: false,
          contentBounds,
          width: newWidth,
          height: newHeight,
          x: currentCenterX - newWidth / 2,
          y: currentCenterY - newHeight / 2,
          isGenerating: false,
          sourcePreviewUrl: undefined,
          remixSuggestions: suggestions,
          generatingPrompt: undefined,
        }
      }));
    } catch (e) {
      console.error(e);
      setImages(prev => prev.filter(img => img.id !== id));
    }
  }, []);

  const generateFromText = useCallback(async (userInput: string, id: number) => {
    try {
        const fullPrompt = TEXT_PROMPT_TEMPLATE(userInput);
        const { imageUrl } = await generateImageFromText(fullPrompt);
        if (!imageUrl) throw new Error("Generation failed, no image returned.");

        const originalImg = new Image();
        originalImg.src = imageUrl;
        originalImageCache.current[id] = originalImg;
      
        const { transparentImage, contentBounds } = await processImageForTransparency(imageUrl);
        const aspectRatio = transparentImage.width / transparentImage.height;
        
        const imageFileForSuggestions = await imageElementToFile(transparentImage, 'suggestion-source.png');
        const suggestions = await getRemixSuggestions(imageFileForSuggestions, REMIX_SUGGESTION_PROMPT);

        setImages(prev => prev.map(img => {
            if (img.id !== id) return img;

            const newWidth = IMAGE_WIDTH;
            const newHeight = IMAGE_WIDTH / aspectRatio;
            const currentCenterX = img.x + img.width / 2;
            const currentCenterY = img.y + img.height / 2;

            return {
                ...img,
                processedImage: transparentImage,
                originalImageUrl: imageUrl,
                showOriginal: false,
                contentBounds,
                width: newWidth,
                height: newHeight,
                x: currentCenterX - newWidth / 2,
                y: currentCenterY - newHeight / 2,
                isGenerating: false,
                remixSuggestions: suggestions,
            }
        }));
    } catch(e) {
        console.error(e);
        setImages(prev => prev.filter(img => img.id !== id));
    }
  }, []);
  
  const addImageToCanvas = useCallback(async (file: File, customPosition?: { x: number; y: number }) => {
    await ensureAudioContext();
    synth.triggerAttackRelease('C4', '8n');

    const id = nextId.current++;
    const canvas = canvasRef.current;
    
    const dropScreenX = customPosition?.x ?? (canvas ? canvas.width / 2 : window.innerWidth / 2);
    const dropScreenY = customPosition?.y ?? (canvas ? canvas.height / 2 : window.innerHeight / 2);
    
    const dropWorldX = (dropScreenX - viewTransform.offsetX) / viewTransform.scale;
    const dropWorldY = (dropScreenY - viewTransform.offsetY) / viewTransform.scale;
    
    const sourcePreviewUrl = URL.createObjectURL(file);
    const previewImage = new Image();
    previewImage.src = sourcePreviewUrl;
    previewImageCache.current[id] = previewImage;
    
    const PLACEHOLDER_WIDTH = 250;

    const newImage: ProcessedImage = {
        id,
        sourceFile: file,
        processedImage: null,
        x: dropWorldX - PLACEHOLDER_WIDTH / 2,
        y: dropWorldY - 100,
        width: PLACEHOLDER_WIDTH,
        height: 200,
        isGenerating: true,
        contentBounds: { x: 0, y: 0, width: PLACEHOLDER_WIDTH, height: 200 },
        sourcePreviewUrl,
        flippedHorizontally: false,
        isVariation: false,
    };
    setImages(prev => [...prev, newImage]);
    generateFromImage(file, id, IMAGE_PROMPT);
  }, [generateFromImage, viewTransform]);


  const handleDrop = useCallback(async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        const rect = canvasRef.current?.getBoundingClientRect();
        const dropX = e.clientX - (rect?.left ?? 0);
        const dropY = e.clientY - (rect?.top ?? 0);
        addImageToCanvas(file, { x: dropX, y: dropY });
      }
    }
  }, [addImageToCanvas]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
        const file = e.target.files[0];
        if (file.type.startsWith('image/')) {
            addImageToCanvas(file);
        }
        e.target.value = '';
    }
  };


  useEffect(() => {
    const handlePaste = async (event: ClipboardEvent) => {
      const items = event.clipboardData?.items;
      if (!items) return;

      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) {
            addImageToCanvas(file);
            event.preventDefault();
            return;
          }
        }
      }
    };

    document.addEventListener('paste', handlePaste);
    return () => {
      document.removeEventListener('paste', handlePaste);
    };
  }, [addImageToCanvas]);

  const getImageAtPosition = useCallback((screenX: number, screenY: number): ProcessedImage | null => {
    const worldX = (screenX - viewTransform.offsetX) / viewTransform.scale;
    const worldY = (screenY - viewTransform.offsetY) / viewTransform.scale;

    for (let i = images.length - 1; i >= 0; i--) {
      const img = images[i];
      if (worldX >= img.x && worldX <= img.x + img.width && worldY >= img.y && worldY <= img.y + img.height) {
        if (img.isGenerating) {
            return img;
        }

        if(img.processedImage && img.contentBounds) {
            const localX = worldX - img.x;
            const localY = worldY - img.y;
            
            const scaleX = img.width / img.processedImage.width;
            const scaleY = img.height / img.processedImage.height;
            
            const scaledBounds = {
                x: img.contentBounds.x * scaleX,
                y: img.contentBounds.y * scaleY,
                width: img.contentBounds.width * scaleX,
                height: img.contentBounds.height * scaleY
            };

            if (localX >= scaledBounds.x && localX <= scaledBounds.x + scaledBounds.width &&
                localY >= scaledBounds.y && localY <= scaledBounds.y + scaledBounds.height) {
                return img;
            }
        }
      }
    }
    return null;
  }, [images, viewTransform]);
  
  const handleInteractionStart = (clientX: number, clientY: number) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const targetImage = getImageAtPosition(x, y);

    if (targetImage) {
        setSelectedImageId(targetImage.id);
        const worldX = (x - viewTransform.offsetX) / viewTransform.scale;
        const worldY = (y - viewTransform.offsetY) / viewTransform.scale;
        setDraggingImage({ id: targetImage.id, offsetX: worldX - targetImage.x, offsetY: worldY - targetImage.y });
        setImages(prev => [...prev.filter(img => img.id !== targetImage.id), targetImage]);
    } else {
        setSelectedImageId(null);
        setIsPanning(true);
        setPanStart({ x: clientX, y: clientY });
    }
  };

  const handleInteractionMove = (clientX: number, clientY: number) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    
    if (draggingImage) {
      const worldX = (x - viewTransform.offsetX) / viewTransform.scale;
      const worldY = (y - viewTransform.offsetY) / viewTransform.scale;
      setImages(prev => prev.map(img => img.id === draggingImage.id ? {
        ...img,
        x: worldX - draggingImage.offsetX,
        y: worldY - draggingImage.offsetY,
      } : img));
    } else if (isPanning) {
        const dx = clientX - panStart.x;
        const dy = clientY - panStart.y;
        setViewTransform(prev => ({ ...prev, offsetX: prev.offsetX + dx, offsetY: prev.offsetY + dy }));
        setPanStart({ x: clientX, y: clientY });
    } else {
        const targetImage = getImageAtPosition(x, y);
        setHoveredImageId(targetImage && !targetImage.isGenerating ? targetImage.id : null);
    }
  };

  const handleInteractionEnd = () => {
    setDraggingImage(null);
    setIsPanning(false);
  };
  
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const targetImage = getImageAtPosition(x, y);

    if (targetImage && !targetImage.isGenerating) {
        const SCALE_SPEED = 0.1;
        const scaleDirection = e.deltaY < 0 ? 1 : -1;
        const scaleFactor = 1 + scaleDirection * SCALE_SPEED;

        setImages(prev => prev.map(img => {
            if (img.id !== targetImage.id) return img;

            const aspectRatio = img.width / img.height;
            let newWidth = img.width * scaleFactor;

            if (newWidth < MIN_WIDTH) newWidth = MIN_WIDTH;
            if (newWidth > MAX_WIDTH) newWidth = MAX_WIDTH;
            
            const newHeight = newWidth / aspectRatio;
            
            const centerX = img.x + img.width / 2;
            const centerY = img.y + img.height / 2;

            const newX = centerX - newWidth / 2;
            const newY = centerY - newHeight / 2;

            return { ...img, width: newWidth, height: newHeight, x: newX, y: newY };
        }));
    } else if (!draggingImage && !isPanning) { // Canvas zoom
        const scaleAmount = 1.1;
        const currentScale = viewTransform.scale;
        
        const newScaleUnclamped = e.deltaY < 0 ? currentScale * scaleAmount : currentScale / scaleAmount;
        const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScaleUnclamped));

        const worldX = (x - viewTransform.offsetX) / currentScale;
        const worldY = (y - viewTransform.offsetY) / currentScale;

        const newOffsetX = x - worldX * newScale;
        const newOffsetY = y - worldY * newScale;

        setViewTransform({ scale: newScale, offsetX: newOffsetX, offsetY: newOffsetY });
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => handleInteractionStart(e.clientX, e.clientY);
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => handleInteractionMove(e.clientX, e.clientY);
  const handleMouseUp = () => handleInteractionEnd();
  
  const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    if (e.touches.length === 1 && !pinchState) {
        handleInteractionStart(e.touches[0].clientX, e.touches[0].clientY);
    } else if (e.touches.length === 2) {
        setDraggingImage(null);

        const t1 = e.touches[0];
        const t2 = e.touches[1];
        const midX = (t1.clientX + t2.clientX) / 2 - rect.left;
        const midY = (t1.clientY + t2.clientY) / 2 - rect.top;
        const distance = Math.hypot(t1.clientX - t2.clientX, t1.clientY - t2.clientY);

        const targetImage = getImageAtPosition(midX, midY);
        if (targetImage && !targetImage.isGenerating) {
            setPinchState({
                id: targetImage.id,
                initialDistance: distance,
                initialWidth: targetImage.width,
                initialHeight: targetImage.height,
                centerX: targetImage.x + targetImage.width / 2,
                centerY: targetImage.y + targetImage.height / 2,
            });
            setImages(prev => [...prev.filter(img => img.id !== targetImage.id), targetImage]);
        } else { // Canvas pinch
            setPinchState({
                id: 'canvas',
                initialDistance: distance,
                initialViewTransform: viewTransform,
                initialMidPoint: { x: midX, y: midY },
            });
        }
    }
  };
  const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
      e.preventDefault();

      if (e.touches.length === 1 && !pinchState) {
          handleInteractionMove(e.touches[0].clientX, e.touches[0].clientY);
      } else if (e.touches.length === 2 && pinchState) {
          const t1 = e.touches[0];
          const t2 = e.touches[1];
          const newDistance = Math.hypot(t1.clientX - t2.clientX, t1.clientY - t2.clientY);

          if (pinchState.id !== 'canvas') { // Object pinch
            const scale = newDistance / pinchState.initialDistance;
            setImages(prev => prev.map(img => {
                if (img.id !== pinchState.id) return img;
                let newWidth = pinchState.initialWidth! * scale;
                if (newWidth < MIN_WIDTH) newWidth = MIN_WIDTH;
                if (newWidth > MAX_WIDTH) newWidth = MAX_WIDTH;
                const newHeight = (pinchState.initialHeight! / pinchState.initialWidth!) * newWidth;
                const newX = pinchState.centerX! - newWidth / 2;
                const newY = pinchState.centerY! - newHeight / 2;
                return { ...img, width: newWidth, height: newHeight, x: newX, y: newY };
            }));
          } else { // Canvas pinch
            const rect = canvasRef.current?.getBoundingClientRect();
            if (!rect || !pinchState.initialViewTransform || !pinchState.initialMidPoint) return;
            const newMidPoint = { x: (t1.clientX + t2.clientX) / 2 - rect.left, y: (t1.clientY + t2.clientY) / 2 - rect.top };
            
            const scaleRatio = newDistance / pinchState.initialDistance;
            const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, pinchState.initialViewTransform.scale * scaleRatio));

            const panDX = newMidPoint.x - pinchState.initialMidPoint.x;
            const panDY = newMidPoint.y - pinchState.initialMidPoint.y;

            const newOffsetX = pinchState.initialViewTransform.offsetX - (pinchState.initialMidPoint.x - pinchState.initialViewTransform.offsetX) * (newScale / pinchState.initialViewTransform.scale - 1) + panDX;
            const newOffsetY = pinchState.initialViewTransform.offsetY - (pinchState.initialMidPoint.y - pinchState.initialViewTransform.offsetY) * (newScale / pinchState.initialViewTransform.scale - 1) + panDY;
            
            setViewTransform({ scale: newScale, offsetX: newOffsetX, offsetY: newOffsetY });
          }
      }
  };
  const handleTouchEnd = (e: React.TouchEvent<HTMLCanvasElement>) => {
      handleInteractionEnd();
      if (e.touches.length < 2) {
          setPinchState(null);
      }
  };

  const handleTextSubmit = async (e: React.FormEvent | React.MouseEvent | React.KeyboardEvent) => {
    e.preventDefault();
    const trimmedInput = textInput.trim();
    if (trimmedInput === '') return;
    
    await ensureAudioContext();
    synth.triggerAttackRelease('C4', '8n');

    const id = nextId.current++;
    const canvas = canvasRef.current;
    
    const dropScreenX = canvas ? canvas.width / 2 : window.innerWidth / 2;
    const dropScreenY = canvas ? canvas.height / 2 : window.innerHeight / 2;
    const dropWorldX = (dropScreenX - viewTransform.offsetX) / viewTransform.scale;
    const dropWorldY = (dropScreenY - viewTransform.offsetY) / viewTransform.scale;
    
    const PLACEHOLDER_WIDTH = 250;

    const newImage: ProcessedImage = {
        id,
        sourceText: trimmedInput,
        processedImage: null,
        x: dropWorldX - PLACEHOLDER_WIDTH / 2,
        y: dropWorldY - PLACEHOLDER_WIDTH / 2,
        width: PLACEHOLDER_WIDTH,
        height: PLACEHOLDER_WIDTH,
        isGenerating: true,
        contentBounds: { x: 0, y: 0, width: PLACEHOLDER_WIDTH, height: PLACEHOLDER_WIDTH },
        flippedHorizontally: false,
        isVariation: false,
    };

    setImages(prev => [...prev, newImage]);
    generateFromText(trimmedInput, id);
    setTextInput('');
  };

  const handleDeleteSelected = async () => {
    if (isActionDisabled || !selectedImage) return;
    await ensureAudioContext();
    synth.triggerAttackRelease('A4', '8n');
    setImages(prev => prev.filter(img => img.id !== selectedImageId));
    setSelectedImageId(null);
  };

  const handleRegenerateSelected = () => {
      if (isActionDisabled || !selectedImage) return;
      const imageToRegen = images.find(img => img.id === selectedImageId);
      if(imageToRegen) {
          setImages(prev => prev.map(img => img.id === selectedImageId ? {...img, isGenerating: true, generatingPrompt: undefined } : img));
          
          if (imageToRegen.sourceFile) {
              generateFromImage(imageToRegen.sourceFile, imageToRegen.id, IMAGE_PROMPT);
          } else if (imageToRegen.sourceText) {
              generateFromText(imageToRegen.sourceText, imageToRegen.id);
          }
      }
  };

  const handleFlipSelected = async () => {
    if (isActionDisabled || !selectedImage) return;
    const imageToFlip = images.find(img => img.id === selectedImageId);
    if (imageToFlip) {
        await ensureAudioContext();
        synth.triggerAttackRelease('G5', '8n');
        setImages(prev => prev.map(img =>
            img.id === selectedImageId
                ? { ...img, flippedHorizontally: !img.flippedHorizontally }
                : img
        ));
    }
  };
  
  const handleDuplicateSelected = async () => {
    if (isActionDisabled || !selectedImage) return;
    const imageToDuplicate = images.find(img => img.id === selectedImageId);
    if (imageToDuplicate) {
        await ensureAudioContext();
        synth.triggerAttackRelease('C4', '8n');

        const newId = nextId.current++;
        const duplicatedImage: ProcessedImage = {
            ...imageToDuplicate,
            id: newId,
            x: imageToDuplicate.x + 40,
            y: imageToDuplicate.y + 20,
        };
        
        // Also duplicate the entry in the original image cache
        const originalImage = originalImageCache.current[imageToDuplicate.id];
        if (originalImage) {
            originalImageCache.current[newId] = originalImage;
        }

        setImages(prev => [...prev, duplicatedImage]);
        setSelectedImageId(newId);
    }
  };

  const handleDownloadSelected = async () => {
    if (isActionDisabled || !selectedImage) return;

    await ensureAudioContext();
    synth.triggerAttackRelease('E5', '8n');

    const imageToDownload = selectedImage.showOriginal
        ? originalImageCache.current[selectedImage.id]
        : selectedImage.processedImage;
    
    const fileName = selectedImage.showOriginal
        ? `voxel-original-${selectedImage.id}.png`
        : `voxel-transparent-${selectedImage.id}.png`;

    if (!imageToDownload || imageToDownload.naturalWidth === 0) {
        console.error("Image source for download is not available or not loaded.");
        return;
    }
    
    const canvas = document.createElement('canvas');
    const aspectRatio = imageToDownload.naturalWidth / imageToDownload.naturalHeight;
    canvas.width = MAX_WIDTH;
    canvas.height = MAX_WIDTH / aspectRatio;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(imageToDownload, 0, 0, canvas.width, canvas.height);

    const link = document.createElement('a');
    link.download = fileName;
    link.href = canvas.toDataURL('image/png');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleRemixSubmit = async (prompt: string) => {
    if (isActionDisabled || !selectedImage) return;
    const imageToRemix = images.find(img => img.id === selectedImageId);

    if (imageToRemix && imageToRemix.processedImage) {
        await ensureAudioContext();
        synth.triggerAttackRelease('E4', '8n');

        const newId = nextId.current++;
        
        const sourceFileForRemix = await imageElementToFile(imageToRemix.processedImage, `remix_of_${imageToRemix.id}.png`);

        const newImage: ProcessedImage = {
            id: newId,
            sourceFile: sourceFileForRemix,
            processedImage: imageToRemix.processedImage, // Show previous image while loading
            x: imageToRemix.x + 40,
            y: imageToRemix.y + 20,
            width: imageToRemix.width,
            height: imageToRemix.height,
            isGenerating: true,
            contentBounds: { x: 0, y: 0, width: imageToRemix.width, height: imageToRemix.height },
            flippedHorizontally: imageToRemix.flippedHorizontally,
            isVariation: true,
            generatingPrompt: prompt,
        };
        setImages(prev => [...prev, newImage]);
        setSelectedImageId(newId);

        const remixPrompt = REMIX_PROMPT_TEMPLATE(prompt);
        generateFromImage(sourceFileForRemix, newId, remixPrompt);
    }
};

const handleRemixKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        const trimmedInput = remixInput.trim();
        if (trimmedInput) {
            handleRemixSubmit(trimmedInput);
            setRemixInput('');
        }
    }
};

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
        if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') {
            return;
        }

        if (e.key === 'm' && selectedImageId !== null) {
            e.preventDefault();
            setImages(prev => prev.map(img =>
                img.id === selectedImageId && img.originalImageUrl
                    ? { ...img, showOriginal: !img.showOriginal }
                    : img
            ));
            return;
        }

        if (selectedImageId === null || isActionDisabled) return;

        // --- MOVEMENT LOGIC ---
        const arrowKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
        if (arrowKeys.includes(e.key)) {
            const selectedImg = images.find(img => img.id === selectedImageId);
            if (!selectedImg) return;
            
            e.preventDefault();

            (async () => {
                await ensureAudioContext();
                synth.triggerAttackRelease('G4', '8n');
            })();

            let newX = selectedImg.x;
            let newY = selectedImg.y;
            
            const moveStep = MOVE_AMOUNT / viewTransform.scale;

            switch (e.key) {
                case 'ArrowUp':    newY -= moveStep; break;
                case 'ArrowDown':  newY += moveStep; break;
                case 'ArrowLeft':  newX -= moveStep; break;
                case 'ArrowRight': newX += moveStep; break;
            }
            
            setImages(prev => prev.map(img => 
                img.id === selectedImageId ? { ...img, x: newX, y: newY } : img
            ));
            return;
        }
        
        // --- OTHER ACTIONS ---
        switch (e.key) {
            case 'Delete':
            case 'Backspace':
                handleDeleteSelected();
                break;
            case 'r':
                handleRegenerateSelected();
                break;
            case 'f':
                handleFlipSelected();
                break;
            case 'd':
                handleDuplicateSelected();
                break;
            default:
                break;
        }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
        window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedImageId, images, isActionDisabled, remixInput, selectedImage, viewTransform.scale]);


  const handleResetCanvas = () => {
    setImages([]);
    setViewTransform({ scale: 1, offsetX: 0, offsetY: 0 });
    setShowResetConfirm(false);
  };

  const handleDownloadCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const link = document.createElement('a');
      link.download = 'banana-world.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
  };

  const actionBarPosition = useMemo(() => {
    if (!selectedImage) return null;

    const actionBarWorldX = selectedImage.x + selectedImage.width / 2;
    const ringRadiusWorld = (Math.min(selectedImage.width, selectedImage.height) / 2 * 0.8);
    const actionBarWorldY = selectedImage.y + selectedImage.height / 2 + ringRadiusWorld + (15 / viewTransform.scale);
    
    const actionBarScreenX = actionBarWorldX * viewTransform.scale + viewTransform.offsetX;
    const actionBarScreenY = actionBarWorldY * viewTransform.scale + viewTransform.offsetY;

    return {
        top: `${actionBarScreenY}px`,
        left: `${actionBarScreenX}px`,
    };
  }, [selectedImage, viewTransform]);


  return (
    <div
      className="w-screen h-screen bg-white"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
        <canvas
            ref={canvasRef}
            style={{ 
              cursor: isPanning ? 'grabbing' : (draggingImage ? 'grabbing' : (hoveredImageId ? 'grab' : 'default'))
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
            onTouchCancel={handleTouchEnd}
            onWheel={handleWheel}
        />
        <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            className="hidden"
            accept="image/*"
        />
        <div className="absolute top-4 left-4 flex items-center gap-2 pointer-events-none">
            <p className="text-lg text-black">BANANA WORLD v0.2</p>
            <button 
                onClick={() => setShowHelpModal(true)}
                className="pointer-events-auto w-6 h-6 flex items-center justify-center border border-black rounded-full text-black bg-white/80"
                aria-label="Show help"
            >
                ?
            </button>
        </div>


        {showResetConfirm && (
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50">
                <div className="bg-white p-8 border border-black shadow-lg flex flex-col items-center gap-4">
                    <p className="text-lg font-mono text-black">Reset world?</p>
                    <div className="flex gap-4">
                        <button onClick={handleResetCanvas} className="px-4 py-2 border border-black bg-black text-white font-mono text-sm hover:bg-neutral-800 transition-colors">Confirm</button>
                        <button onClick={() => setShowResetConfirm(false)} className="px-4 py-2 border border-black bg-white text-black font-mono text-sm transition-colors">Cancel</button>
                    </div>
                </div>
            </div>
        )}

        {showHelpModal && (
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowHelpModal(false)}>
                <div className="bg-white p-6 border border-black shadow-lg font-mono text-black text-sm flex flex-col gap-4 max-w-sm">
                    <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2">
                        <span className="font-bold text-right">R:</span><span>Regenerate</span>
                        <span className="font-bold text-right">F:</span><span>Flip</span>
                        <span className="font-bold text-right">D:</span><span>Duplicate</span>
                        <span className="font-bold text-right">M:</span><span>Show unmasked image</span>
                        <span className="font-bold text-right">Backspace:</span><span>Delete</span>
                        <span className="font-bold text-right">Arrow keys:</span><span>Move object</span>
                        <span className="font-bold text-right">Mouse Wheel:</span><span>Scale object / Zoom view</span>
                        <span className="font-bold text-right">Pinch:</span><span>Scale object / Zoom view</span>
                    </div>
                    <div className="text-xs text-gray-500 text-center pt-4 border-t border-gray-200">
                        Built with Gemini 2.5 Flash image (nano-banana) by <a href="https://x.com/alexanderchen" target="_blank" rel="noopener noreferrer" className="underline hover:text-black transition-colors">@alexanderchen</a> and Remixed by <a href="https://x.com/drashyakuruwa" target="_blank" rel="noopener noreferrer" className="underline hover:text-black transition-colors">@drashyakuruwa</a>
                    </div>
                </div>
            </div>
        )}

        {selectedImage && !selectedImage.isGenerating && actionBarPosition && (
            <div
                className="absolute flex flex-col items-center gap-2"
                style={{
                    ...actionBarPosition,
                    transform: 'translateX(-50%)',
                    zIndex: 10,
                }}
            >
                <div className="flex flex-nowrap bg-white border border-black">
                    <button
                        onClick={handleRegenerateSelected}
                        disabled={isActionDisabled}
                        className="h-10 w-10 p-2 box-border text-black disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors hover:bg-gray-100 border-r border-black"
                        aria-label="Regenerate selected object"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M3 2v6h6"/>
                            <path d="M21 12A9 9 0 0 0 6 5.3L3 8"/>
                            <path d="M21 22v-6h-6"/>
                            <path d="M3 12a9 9 0 0 0 15 6.7l3-2.7"/>
                        </svg>
                    </button>
                    <button
                        onClick={handleFlipSelected}
                        disabled={isActionDisabled}
                        className="h-10 w-10 p-2 box-border text-black disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors hover:bg-gray-100 border-r border-black"
                        aria-label="Flip selected object"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m3 7 5 5-5 5V7"/><path d="m21 7-5 5 5 5V7"/><path d="M12 20v-2M12 16v-2M12 12V3"/></svg>
                    </button>
                    <button
                        onClick={handleDuplicateSelected}
                        disabled={isActionDisabled}
                        className="h-10 w-10 p-2 box-border text-black disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors hover:bg-gray-100 border-r border-black"
                        aria-label="Duplicate selected object"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    </button>
                    <button
                        onClick={handleDeleteSelected}
                        disabled={isActionDisabled}
                        className="h-10 w-10 p-2 box-border text-black disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors hover:bg-gray-100 border-r border-black"
                        aria-label="Delete selected object"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                    </button>
                    <button
                        onClick={handleDownloadSelected}
                        disabled={isActionDisabled}
                        className="h-10 w-10 p-2 box-border text-black disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors hover:bg-gray-100"
                        aria-label="Download selected object"
                    >
                       <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                    </button>
                </div>

                <div className="flex flex-col items-start">
                    <div className="w-80 relative">
                         <input
                            ref={remixInputRef}
                            type="text"
                            value={remixInput}
                            onChange={(e) => setRemixInput(e.target.value)}
                            onKeyDown={handleRemixKeyDown}
                            placeholder="Remix this ..."
                            className="w-full h-10 box-border px-3 py-2 border border-black bg-white text-black text-sm placeholder-neutral-600 focus:outline-none"
                        />
                    </div>
                    {selectedImage.remixSuggestions && selectedImage.remixSuggestions.length > 0 && (
                        <button
                            onClick={() => {
                                const suggestion = selectedImage.remixSuggestions?.[suggestionIndex];
                                if (suggestion) handleRemixSubmit(suggestion);
                            }}
                            className="w-auto max-w-full text-xs text-left text-gray-600 hover:text-black transition-all duration-300 ease-in-out cursor-pointer p-2 bg-white/90 border border-black -mt-px"
                            title="Click to try this suggestion"
                        >
                            💡 {selectedImage.remixSuggestions[suggestionIndex]}
                        </button>
                    )}
                </div>
            </div>
        )}
       
        <div className="absolute bottom-4 left-4 right-4 flex flex-col md:flex-row md:justify-between md:items-center gap-2">
            {/* Top Row (Mobile) / Left Side (Desktop) */}
            <div className="w-full md:max-w-lg flex items-center gap-2">
                <div className="w-full flex-grow relative">
                    <input
                        type="text"
                        value={textInput}
                        onChange={(e) => setTextInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                                e.preventDefault();
                                handleTextSubmit(e);
                            }
                        }}
                        placeholder="Create anything ..."
                        className="w-full h-12 box-border pl-4 pr-12 py-3 border border-black bg-white/80 text-black text-sm placeholder-neutral-600 focus:outline-none"
                        aria-label="Prompt input"
                    />
                    <button
                        type="button"
                        onClick={handleTextSubmit}
                        className="absolute right-3 p-1 text-black transition-transform hover:scale-110 top-1/2 -translate-y-1/2"
                        aria-label="Submit prompt"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                    </button>
                </div>
                <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="h-12 w-12 p-3 box-border border border-black bg-white/80 text-black flex-shrink-0 transition-colors hover:bg-gray-100 flex items-center justify-center"
                    aria-label="Upload image"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
                </button>
            </div>

            {/* Bottom Row (Mobile) / Right Side (Desktop) */}
            <div className="flex flex-nowrap justify-center md:justify-end gap-2 overflow-x-auto">
                <button 
                    onClick={() => setShowResetConfirm(true)} 
                    className="h-12 box-border flex-shrink-0 px-4 py-3 border border-black bg-white/80 text-black text-sm flex items-center justify-center gap-1 transition-colors hover:bg-gray-100"
                    aria-label="Reset canvas"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    Reset
                </button>
                <button
                    onClick={handleDownloadCanvas}
                    className="h-12 box-border flex-shrink-0 px-4 py-3 border border-black bg-white/80 text-black text-sm flex items-center justify-center gap-1 transition-colors hover:bg-gray-100"
                    aria-label="Download canvas"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                    Save
                </button>
            </div>
        </div>
    </div>
  );
};

export default App;