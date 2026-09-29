import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

  app.use(express.json({ limit: '10mb' }));

  // Initialize Gemini SDK with server-side environment key
  const ai = new GoogleGenAI(
    process.env.GEMINI_API_KEY ? { apiKey: process.env.GEMINI_API_KEY } : {}
  );

  // 1. Music Generation via Lyria (lyria-3-clip-preview / lyria-3-pro-preview)
  app.post('/api/music/generate', async (req: Request, res: Response) => {
    try {
      const {
        prompt,
        model = 'lyria-3-clip-preview',
        style = '1980s Retro Bollywood Synth & Cassette Tape',
      } = req.body;

      if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required' });
      }

      // Valid models for Lyria
      const validModel =
        model === 'lyria-3-pro-preview'
          ? 'lyria-3-pro-preview'
          : 'lyria-3-clip-preview';

      const enrichedPrompt = `Generate a nostalgic retro 1980s or 1990s vintage Indian music track in the style of: ${style}. Atmospheric, warm tape saturation, analog synthesizer, dholak/tabla rhythm and nostalgic cassette boombox fidelity. User prompt: ${prompt}`;

      const response = await ai.models.generateContentStream({
        model: validModel,
        contents: enrichedPrompt,
      });

      let audioBase64 = '';
      let lyrics = '';
      let mimeType = 'audio/wav';

      for await (const chunk of response) {
        const parts = chunk.candidates?.[0]?.content?.parts;
        if (!parts) continue;
        for (const part of parts) {
          if (part.inlineData?.data) {
            if (!audioBase64 && part.inlineData.mimeType) {
              mimeType = part.inlineData.mimeType;
            }
            audioBase64 += part.inlineData.data;
          }
          if (part.text && !lyrics) {
            lyrics = part.text;
          }
        }
      }

      if (!audioBase64) {
        return res.status(500).json({
          error:
            'Music model did not return audio data. Ensure a valid paid API key is selected in AI Studio.',
        });
      }

      return res.json({
        success: true,
        model: validModel,
        mimeType,
        audioBase64,
        lyrics: lyrics || 'Instrumental analog cassette track',
        title: prompt.slice(0, 50),
      });
    } catch (err: unknown) {
      console.error('Error generating music with Lyria:', err);
      const message = err instanceof Error ? err.message : String(err);
      return res.status(500).json({
        error: message || 'Failed to generate music track with Lyria.',
      });
    }
  });

  // 2. Google Search Grounding with gemini-3.5-flash
  app.post('/api/research/search', async (req: Request, res: Response) => {
    try {
      const { query, topic = '1980s Indian Transit & Street Culture' } = req.body;

      if (!query) {
        return res.status(400).json({ error: 'Search query is required' });
      }

      const prompt = `You are an expert cultural and historical archivist of late 1980s and 1990s urban India (transit, everyday streets, audio culture). 
Use Google Search grounding to retrieve accurate, verified, and detailed historical facts regarding: "${query}".
Focus on:
1. Historical details and mechanical specifications (e.g. vintage Ashok Leyland / Tata buses, Bajaj auto-rickshaws, Lambretta/Bajaj Chetak scooters).
2. Everyday street reality and audio culture (T-Series audio cassettes, roadside tape stalls, Weston / Murphy two-speaker boomboxes, bus conductor whistle and bell cords).
3. Authentic cultural context from the 1980s/1990s era.

Format response clearly with key findings, verified historical milestones, and visual memory details.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: prompt,
        config: {
          tools: [{ googleSearch: {} }],
        },
      });

      const summaryText = response.text || '';
      const groundingChunks =
        response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
      const webSearchQueries =
        response.candidates?.[0]?.groundingMetadata?.webSearchQueries || [];

      // Extract search sources and URLs
      const sources: { title: string; uri: string }[] = [];
      for (const chunk of groundingChunks) {
        if (chunk.web?.uri) {
          sources.push({
            title: chunk.web.title || chunk.web.uri,
            uri: chunk.web.uri,
          });
        }
      }

      return res.json({
        success: true,
        query,
        topic,
        summary: summaryText,
        sources,
        webSearchQueries,
      });
    } catch (err: unknown) {
      console.error('Error with Google Search Grounding:', err);
      const message = err instanceof Error ? err.message : String(err);
      return res.status(500).json({
        error: message || 'Failed to execute grounded search.',
      });
    }
  });

  // Dev server with Vite middleware or static production serving
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
