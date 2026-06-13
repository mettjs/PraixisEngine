up:
	docker compose up --build

up-local:
	docker compose -f docker-compose.yml -f docker-compose.local.yml up --build

up-chroma:
	docker compose -f docker-compose.yml -f docker-compose.chroma.yml up --build

down:
	docker compose down

down-local:
	docker compose -f docker-compose.yml -f docker-compose.local.yml down

down-chroma:
	docker compose -f docker-compose.yml -f docker-compose.chroma.yml down
