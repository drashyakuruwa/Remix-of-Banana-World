import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Modality, Type } from '@google/genai';

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '50mb' }));

  app.post('/api/gemini/generate-image', async (req, res) => {
    try {
      const { prompt, base64ImageData, mimeType } = req.body;
      const ai = new GoogleGenAI({ 
        apiKey: process.env.GEMINI_API_KEY,
        httpOptions: { headers: { 'User-Agent': 'aistudio-build' } }
      });
      
      const parts: any[] = [];
      if (base64ImageData && mimeType) {
        parts.push({ inlineData: { data: base64ImageData, mimeType } });
      }
      parts.push({ text: prompt });

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts },
        config: {
          responseModalities: [Modality.IMAGE, Modality.TEXT],
        },
      });

      let imageUrl: string | null = null;
      let text: string | null = null;

      if (response.candidates && response.candidates.length > 0) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData && part.inlineData.data) {
            imageUrl = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
          } else if (part.text) {
            text = part.text;
          }
        }
      }

      if (!imageUrl) {
        return res.status(500).json({ error: text || 'No image generated in the API response' });
      }

      res.json({ imageUrl, text });
    } catch (e: any) {
      console.error("Generate content error:", e);
      res.status(500).json({ error: e.message || "Unknown error generating image" });
    }
  });

  app.post('/api/gemini/remix-suggestions', async (req, res) => {
    try {
      const { prompt, base64ImageData, mimeType } = req.body;
      const ai = new GoogleGenAI({ 
        apiKey: process.env.GEMINI_API_KEY,
        httpOptions: { headers: { 'User-Agent': 'aistudio-build' } }
      });
      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: {
          parts: [
            { text: prompt },
            { inlineData: { data: base64ImageData, mimeType } },
          ],
        },
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              prompts: {
                type: Type.ARRAY,
                items: {
                  type: Type.STRING,
                  description: 'A short remix prompt idea.',
                },
              },
            },
            required: ['prompts'],
          },
        },
      });

      const jsonText = response.text?.trim() || "";
      let result = { prompts: [] };
      if (jsonText) {
          try {
              result = JSON.parse(jsonText);
          } catch(e) { /* ignore */ }
      }
      res.json({ prompts: result.prompts?.slice(0, 5) || [] });
    } catch (e: any) {
      console.error("Remix suggestions error:", e);
      res.status(500).json({ error: e.message });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*all', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
