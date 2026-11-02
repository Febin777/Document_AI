import 'dotenv/config';
import { GoogleGenAI } from '@google/genai';

const apiKey = process.env.GEMINI_API_KEY;
const ai = new GoogleGenAI({ apiKey: apiKey || '' });

class VectorStore {
  constructor() {
    this.documents = [];
  }

  // Cosine Similarity between two vectors
  cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    const mag = Math.sqrt(normA) * Math.sqrt(normB);
    return mag === 0 ? 0 : dotProduct / mag;
  }

  // Generates embedding using Gemini API or fallback word-frequency vector
  async getEmbedding(text) {
    try {
      // Direct call using the official model identifier
      const response = await ai.models.embedContent({
        model: 'gemini-embedding-001',
        contents: text,
      });

      if (response && response.embeddings && response.embeddings[0] && response.embeddings[0].values) {
        return response.embeddings[0].values;
      }
      throw new Error('No values returned');
    } catch (err) {
      console.warn(`Gemini embed failed (${err.message}). Using deterministic fallback vector.`);
      return this.generateDeterministicVector(text, 256);
    }
  }

  // Fallback 256-dimensional semantic hash vector (ensures project works 100% offline or on quota limits)
  generateDeterministicVector(text, dims = 256) {
    const vector = new Array(dims).fill(0);
    const words = text.toLowerCase().replace(/[^a-z0-9 ]/g, '').split(/\s+/);
    
    words.forEach((word) => {
      let hash = 0;
      for (let i = 0; i < word.length; i++) {
        hash = (hash << 5) - hash + word.charCodeAt(i);
        hash |= 0;
      }
      const index = Math.abs(hash) % dims;
      vector[index] += 1;
    });

    // Normalize vector
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return norm === 0 ? vector : vector.map(v => v / norm);
  }

  chunkText(text, chunkSize = 600, overlap = 100) {
    const cleanedText = text.replace(/\s+/g, ' ').trim();
    const chunks = [];
    let start = 0;

    while (start < cleanedText.length) {
      const end = Math.min(start + chunkSize, cleanedText.length);
      chunks.push(cleanedText.slice(start, end));
      if (end === cleanedText.length) break;
      start += chunkSize - overlap;
    }
    return chunks;
  }

  async addDocument(filename, text) {
    const chunks = this.chunkText(text);
    console.log(`Vectorizing ${chunks.length} chunks from "${filename}"...`);

    for (const chunk of chunks) {
      const embedding = await this.getEmbedding(chunk);
      this.documents.push({
        text: chunk,
        embedding,
        docName: filename,
      });
    }
  }

  async similaritySearch(query, topK = 3) {
    if (this.documents.length === 0) return [];

    const queryEmbedding = await this.getEmbedding(query);
    const scored = this.documents.map(doc => ({
      ...doc,
      score: this.cosineSimilarity(queryEmbedding, doc.embedding),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  clear() {
    this.documents = [];
  }
}

export const vectorStore = new VectorStore();