import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, Type } from '@google/genai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const port = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '256kb' }));
app.use('/roadrunner-motion-demo', express.static(__dirname));
app.use(express.static(__dirname));

const aiClient = process.env.GEMINI_API_KEY
  ? new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    })
  : null;

app.post('/api/critic', async (req, res) => {
  if (!aiClient) {
    return res.json({ available: false, reason: 'no_api_key' });
  }

  const metrics = req.body;
  if (!metrics || typeof metrics !== 'object') {
    return res.status(400).json({ available: false, error: 'Invalid metrics payload' });
  }

  try {
    const response = await aiClient.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents: `Analyze these programmatically computed robot simulation validation metrics and issues:\n${JSON.stringify(metrics, null, 2)}`,
      config: {
        systemInstruction:
          'You are a robotics simulation motion critic for Roadrunner (a bipedal wheeled-leg robot). ' +
          'Respond in concise Simplified Chinese. Use ONLY the provided metrics and detected issues; ' +
          'never invent physical measurements, mass, torque, or friction values. ' +
          'Explicitly state uncertainties where parameters are unavailable in the prototype model.',
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: { type: Type.STRING },
            terrainNotes: { type: Type.STRING },
            recommendations: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            uncertainties: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
          },
          required: ['summary', 'terrainNotes', 'recommendations', 'uncertainties'],
        },
      },
    });

    const parsed = JSON.parse((response.text ?? '{}').trim());
    return res.json({
      available: true,
      model: 'gemini-4-hydrogen',
      analysis: parsed,
    });
  } catch (err) {
    console.warn('AI critic endpoint error, falling back to local explanation:', err?.message || err);
    return res.json({ available: false, reason: 'ai_unavailable' });
  }
});

app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Roadrunner Motion Studio listening on http://0.0.0.0:${port}`);
});
