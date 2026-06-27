up:
	docker compose up --build

up-local:
	docker compose -f docker-compose.yml -f docker-compose.local.yml up --build

up-chroma:
	docker compose -f docker-compose.yml -f docker-compose.chroma.yml up --build

up-local-vllm:
	docker compose -f docker-compose.yml -f docker-compose.local.yml -f docker-compose.vllm.yml up --build

up-chroma-vllm:
	docker compose -f docker-compose.yml -f docker-compose.chroma.yml -f docker-compose.vllm.yml up --build

up-local-litellm:
	docker compose -f docker-compose.yml -f docker-compose.local.yml -f docker-compose.litellm.yml up --build

up-chroma-litellm:
	docker compose -f docker-compose.yml -f docker-compose.chroma.yml -f docker-compose.litellm.yml up --build

down:
	docker compose down

down-local:
	docker compose -f docker-compose.yml -f docker-compose.local.yml down

down-chroma:
	docker compose -f docker-compose.yml -f docker-compose.chroma.yml down

down-local-vllm:
	docker compose -f docker-compose.yml -f docker-compose.local.yml -f docker-compose.vllm.yml down

down-chroma-vllm:
	docker compose -f docker-compose.yml -f docker-compose.chroma.yml -f docker-compose.vllm.yml down

down-local-litellm:
	docker compose -f docker-compose.yml -f docker-compose.local.yml -f docker-compose.litellm.yml down

down-chroma-litellm:
	docker compose -f docker-compose.yml -f docker-compose.chroma.yml -f docker-compose.litellm.yml down
