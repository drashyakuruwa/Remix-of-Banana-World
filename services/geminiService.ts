import { GoogleGenAI, Modality, Type } from "@google/genai";

const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            const result = reader.result as string;
            // The result includes the "data:image/jpeg;base64," prefix, which we need to remove.
            resolve(result.split(',')[1]);
        };
        reader.onerror = error => reject(error);
    });
};

export const generateImageWithPrompt = async (imageFile: File, prompt: string): Promise<{ imageUrl: string | null; text: string | null; }> => {
    if (!process.env.API_KEY) {
        throw new Error("API_KEY environment variable not set. Please ensure it's configured.");
    }
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

    const base64ImageData = await fileToBase64(imageFile);
    const mimeType = imageFile.type;

    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image-preview',
        contents: {
            parts: [
                {
                    inlineData: {
                        data: base64ImageData,
                        mimeType: mimeType,
                    },
                },
                {
                    text: prompt,
                },
            ],
        },
        config: {
            responseModalities: [Modality.IMAGE, Modality.TEXT],
        },
    });
    
    let imageUrl: string | null = null;
    let text: string | null = null;
    
    if (response.candidates && response.candidates.length > 0) {
        for (const part of response.candidates[0].content.parts) {
            if (part.inlineData && part.inlineData.data) {
                const base64ImageBytes: string = part.inlineData.data;
                imageUrl = `data:${part.inlineData.mimeType};base64,${base64ImageBytes}`;
            } else if (part.text) {
                text = part.text;
            }
        }
    }

    if (!imageUrl) {
        throw new Error(text || "No image was generated in the API response.");
    }

    return { imageUrl, text };
};


export const generateImageFromText = async (prompt: string): Promise<{ imageUrl: string | null; text: string | null; }> => {
    if (!process.env.API_KEY) {
        throw new Error("API_KEY environment variable not set. Please ensure it's configured.");
    }
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    
    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image-preview',
        contents: {
            parts: [
                {
                    text: prompt,
                },
            ],
        },
        config: {
            responseModalities: [Modality.IMAGE, Modality.TEXT],
        },
    });
    
    let imageUrl: string | null = null;
    let text: string | null = null;
    
    if (response.candidates && response.candidates.length > 0) {
        for (const part of response.candidates[0].content.parts) {
            if (part.inlineData && part.inlineData.data) {
                const base64ImageBytes: string = part.inlineData.data;
                imageUrl = `data:${part.inlineData.mimeType};base64,${base64ImageBytes}`;
            } else if (part.text) {
                text = part.text;
            }
        }
    }

    if (!imageUrl) {
        throw new Error(text || "No image was generated in the API response.");
    }

    return { imageUrl, text };
};

export const getRemixSuggestions = async (imageFile: File, prompt: string): Promise<string[]> => {
    if (!process.env.API_KEY) {
        throw new Error("API_KEY environment variable not set. Please ensure it's configured.");
    }
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const base64ImageData = await fileToBase64(imageFile);

    const textPart = { text: prompt };
    const imagePart = {
        inlineData: {
            mimeType: imageFile.type,
            data: base64ImageData,
        },
    };

    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: { parts: [textPart, imagePart] },
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        prompts: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.STRING,
                                description: "A short remix prompt idea."
                            }
                        }
                    },
                    required: ["prompts"]
                }
            }
        });

        const jsonText = response.text.trim();
        const result = JSON.parse(jsonText);
        
        if (result.prompts && Array.isArray(result.prompts) && result.prompts.length > 0) {
            return result.prompts.slice(0, 5); // Ensure we only return 5
        }
        return [];
    } catch (error) {
        console.error("Error getting remix suggestions:", error);
        return []; // Return empty array on failure
    }
};