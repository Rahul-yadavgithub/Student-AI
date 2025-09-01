import fs from 'fs';
import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';

import { Pinecone } from '@pinecone-database/pinecone';

import { PineconeStore } from '@langchain/pinecone';

async function indexDoc() {
    const PDF_PATH = './dbms.pdf';

    if (!fs.existsSync(PDF_PATH)) {
        console.error("PDF not found at:", PDF_PATH);
        return;
    }

    const pdfloader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfloader.load();

    // Chunking text
    const textsplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });
    const chunkedDocs = await textsplitter.splitDocuments(rawDocs);

    console.log("Chunking Done of PDF");

    // Create embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });

    console.log("Embedding Model configured");

    // Generate embedding vectors for all chunks
    const embeddingVectors = await embeddings.embedDocuments(
        chunkedDocs.map(doc => doc.pageContent)
    );


    // Initialisation of Pinecone 

    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    console.log("Pinecone Configured");

    // now we want to give the infomation to the langchain so that langchain can make all the process in one step
    // Tell them about the chucking which you did in this case 
    // Tell them about the model which you used hear 
    // Tell the Database where you are storing this 
    // Now after getting such infromation langchain can do all the things automatically 

    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
        pineconeIndex,
        maxConcurrency : 5
    });

    console.log("Data is Stored Successfully");


}

// Run the function
indexDoc();
