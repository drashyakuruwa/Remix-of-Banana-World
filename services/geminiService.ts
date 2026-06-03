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
    const base64ImageData = await fileToBase64(imageFile);
    const mimeType = imageFile.type;

    const res = await fetch('/api/gemini/generate-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, base64ImageData, mimeType })
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({ error: 'Network error ' + res.status }));
        throw new Error(err.error || `Failed to generate image: ${res.status}`);
    }

    return await res.json();
};

export const generateImageFromText = async (prompt: string): Promise<{ imageUrl: string | null; text: string | null; }> => {
    const res = await fetch('/api/gemini/generate-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({ error: 'Network error ' + res.status }));
        throw new Error(err.error || `Failed to generate image: ${res.status}`);
    }

    return await res.json();
};

export const getRemixSuggestions = async (imageFile: File, prompt: string): Promise<string[]> => {
    const base64ImageData = await fileToBase64(imageFile);
    const mimeType = imageFile.type;

    try {
        const res = await fetch('/api/gemini/remix-suggestions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, base64ImageData, mimeType })
        });
        
        if (!res.ok) return [];

        const result = await res.json();
        return result.prompts || [];
    } catch (error) {
        console.error("Error getting remix suggestions:", error);
        return [];
    }
};