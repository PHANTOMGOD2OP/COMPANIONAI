import { MemoryManager } from "@/lib/memory";
import db from "@/lib/prismadb";

import { OpenAI } from "langchain/llms/openai";
import { CallbackManager } from "langchain/callbacks";
import { PromptTemplate } from "langchain/prompts";
import { LLMChain } from "langchain/chains";


import { rateLimit } from "@/lib/rate-limit";
import { currentUser } from "@clerk/nextjs";
import { LangChainStream, StreamingTextResponse } from "ai";
import { NextResponse } from "next/server";

export async function POST(req: Request,
    { params }: { params: { chatId: string } }
) {    
    try {

        const { prompt } = await req.json();
        const user = await currentUser();

        if (!user) {
            return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
        }

        const identifier = `${user.id}-${params.chatId}`;
        const { success } = await rateLimit(identifier);

        if (!success) {
            return NextResponse.json({ message: 'Rate limit exceeded' }, { status: 429 });
        }

        // update the companion with the new prompt and return the updated companion
        const companion = await db.companion.update({
            where: {
                id: params.chatId
            },

            data: {
                messages: {
                    create: {
                        content: prompt,
                        role: 'user',
                        userId: user.id,
                    }
                }
            }
        });

        if (!companion) {
            return NextResponse.json({ message: 'Companion not found' }, { status: 404 });
        }

        const companion_id = companion.id;
        const pine_cone_identifier = companion_id + '.pine_cone';

        // let's create companion key for redis
        const companionKey = {
            companionName: companion_id,
            modelName: 'llama2-13b',
            userId: user.id
        };

        const memManager = await MemoryManager.getInstance();

        // if it's the first time talking to the companion, we need to seed the chat history
        const records = await memManager.readLatestHistory(companionKey);
        if (!records || records.length == 0) {
            await memManager.seedChatHistory(companion.seed, '\n\n', companionKey);
        }

        // write the new prompt to the chat history in the format of our seed!!
        await memManager.writeToHistory("User: " + prompt + "\n", companionKey);

        // let's get similar docs from our pinecone db
        const recentChatHistory = await memManager.readLatestHistory(companionKey);

        const similarDocs = await memManager.vectorSearch(
            recentChatHistory,
            pine_cone_identifier
        )

        let relevantHistory = "";
        if (!!similarDocs && similarDocs.length !== 0) {
            relevantHistory = similarDocs.map((doc) => doc.pageContent).join("\n");
        }

        console.log("RELEVANT HISTORY: ", relevantHistory);

        const { handlers } = LangChainStream();

        const model = new OpenAI(
            {
                modelName: "gpt-3.5-turbo-16k",
                openAIApiKey: process.env.OPENAI_API_KEY,
                callbackManager: CallbackManager.fromHandlers(handlers),
            }
        )

        model.verbose = true;
        const chainPrompt = PromptTemplate.fromTemplate(
            `
            ONLY generate plain sentences without prefix of who is speaking. DO NOT use ${companion.name}: prefix. 
            
            You are ${companion.name} and are currently talking to ${currentUser.name}.
            
            ${companion.instructions}
            
            Below are relevant details about ${companion.name}'s past and the conversation you are in.
            ${relevantHistory}
            
            Below is a relevant conversation history
            ${recentChatHistory}\n${companion.name}
            `
        )

        const chain = new LLMChain({
            llm: model,
            prompt: chainPrompt
        })

        const response = await chain
            .call({
                relevantHistory,
                recentChatHistory: recentChatHistory,
            })
            .catch(
                console.error
            )

        const real_response = response!.text;
        
        if (real_response !== undefined && real_response.length > 1) {
            memManager.writeToHistory("" + real_response.trim(), companionKey);

            // upsert it into the pinecone db
            const current_history = `${real_response.trim()}`;
            await memManager.UpsertChatHistory(
                recentChatHistory + "\n" + current_history,
                pine_cone_identifier
            );

            await db.companion.update({
                where: {
                    id: params.chatId
                },
                data: {
                    messages: {
                        create: {
                            content: real_response,
                            role: "system",
                            userId: user.id,
                        },
                    },
                }
            });
        }

        var Readable = require("stream").Readable;
        let s = new Readable();
        s.push(real_response);
        s.push(null);

        return new StreamingTextResponse(s);
    }
    catch (error) {
        console.log('ERROR GETTING RESPONSE FROM CHATGPT: ', error);
        return new NextResponse("Internal server error", { status: 500 });
    }
    }
