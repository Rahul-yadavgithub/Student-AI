import * as dotenv from 'dotenv';
dotenv.config();

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({});

const History = [];

import readlineSync from 'readline-sync';

// Now we do something on the user question 

async function chatting(question){

    // First of all check the relavent meaning of the question from the transformQuery question 

    const queries = await transformQuery(question);

    // convert this question to the vector 

    const embedding = new GoogleGenerativeAIEmbeddings({
        apiKey : process.env.GEMINI_API_KEY,
        model : 'text-embedding-004',
    });

    // Our Query Vector 
    
    const queryVector = await embedding.embedQuery(queries);

    // Now we need to make the connect with the Pinencoe for seraching our query

    const pinecone = new Pinecone();

    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    // Now connection estabalised we can now search for this 

    const SearchResults = await pineconeIndex.query({
      topK : 10,
      vector : queryVector,
      includeMetadata: true,
    });

// Now we need to give the text to the llm for the generation of the text

const context = SearchResults.matches.map(match => match.metadata.text).join("\n\n---\n\n");

// Now we need to give the context to the LLM 

History.push({
    role: 'user',
    parts :[{text: queries}]
});

const response = await ai.models.generateContent({
    model : "gemini-2.0-flash",
    contents: History,
    config :{
        systemInstruction: `You have to behave like a DBMS Expert.
    You will be given a context of relevant information and a user question.
    Your task is to answer the user's question based ONLY on the provided context.
    If the answer is not in the context, you must say "I could not find the answer in the provided document."
    Keep your answers clear, concise, and educational.
      
      Context: ${context}
      `,
    },

});

History.push({
    role: 'model',
    parts : [{text: response.text}],
});

console.log('\n');
console.log(response.text);



}

// Now we want that our history get saved and if in future we ask any question on the basis of history we want to get anwer

async function transformQuery(question){

    History.push({
        role : 'user',
        parts :[{text:question}]
    });

    const response = await ai.models.generateContent({
        model : "gemini-2.0-flash",
        contents :History,
        config :{
            systemInstruction :`You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
                Only output the rewritten question and nothing else.
            `,
        },
    });
    History.pop();
    return response.text;
}



async function main(){
    const userProblem = readlineSync.question("Ask me anyThing --->");
    await chatting(userProblem);
    main();
}

main();