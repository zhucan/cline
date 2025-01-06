import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"
import { ApiHandler } from "../"
import { ApiHandlerOptions, ModelInfo, ZhipuModelId, zhipuDefaultModelId, zhipuModels } from "../../shared/api"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream } from "../transform/stream"

export class ZhipuHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: OpenAI

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.client = new OpenAI({
			baseURL: "https://open.bigmodel.cn/api/paas/v4/chat/completions",
			apiKey: this.options.zhipuApiKey,
		})
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const stream = await this.client.chat.completions.create({
			model: this.getModel().id,
			messages: [{ role: "system", content: systemPrompt }, ...convertToOpenAiMessages(messages)],
			temperature: 0.7,
			max_tokens: this.getModel().info.maxTokens,
			top_p: 0.9,
			stream: true,
		})

		for await (const chunk of stream) {
			const delta = chunk.choices[0]?.delta
			if (delta?.content) {
				yield {
					type: "text",
					text: delta.content,
				}
			}

			if (chunk.usage) {
				yield {
					type: "usage",
					inputTokens: chunk.usage.prompt_tokens || 0,
					outputTokens: chunk.usage.completion_tokens || 0,
				}
			}
		}
	}

	getModel(): { id: ZhipuModelId; info: ModelInfo } {
		const modelId = this.options.apiModelId
		if (modelId && modelId in zhipuModels) {
			const id = modelId as ZhipuModelId
			return { id, info: zhipuModels[id] }
		}
		return { id: zhipuDefaultModelId, info: zhipuModels[zhipuDefaultModelId] }
	}
}
