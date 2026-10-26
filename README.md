# LLM Router

OpenAI-совместимый прокси-роутер на FastAPI. Одна точка входа для всех LLM-провайдеров:
Mistral, Ollama, Groq, NVIDIA NIM, Cerebras — любого API в формате OpenAI.
Если одна модель не ответила (таймаут, 429, 5xx, 401/403) — запрос автоматически
переключается на следующую модель в цепочке.

## Быстрый старт

```bash
pip install -r requirements.txt
nano config.json        # вписать свои ключи/URL/модели
python server.py        # поднимется на http://0.0.0.0:8000
```

Переменные окружения (имеют приоритет над конфигом):
`PORT`, `HOST`, `CONFIG_PATH`.

## Конфигурация (config.json)

Секция `server`:

| Поле | По умолчанию | Что делает |
|------|--------------|------------|
| `host` | `0.0.0.0` | адрес слушателя |
| `port` | `8000` | порт (в облаках задавайте через переменную `PORT`) |
| `api_key` | — | ключ для доступа к самому роутеру |
| `require_auth` | `true` | требовать `Authorization: Bearer <api_key>` на защищённых эндпоинтах |
| `request_timeout` | `120` | таймаут запроса к провайдеру, сек |
| `connect_timeout` | `10` | таймаут установки соединения, сек |
| `max_attempts` | `0` (все) | максимум попыток в цепочке за один запрос |

Секция `providers` — список провайдеров. Каждый:

| Поле | Что делает |
|------|------------|
| `name` | имя (попадает в заголовок ответа `x-router-provider`) |
| `base_url` | базовый URL **с /v1** (Mistral: `https://api.mistral.ai/v1`, Ollama: `http://localhost:11434/v1`) |
| `api_key` | ключ провайдера (для Ollama — любое значение) |
| `models` | список моделей для роутинга |
| `enabled` | `true/false` — участвует ли в цепочке |
| `timeout` | (опц.) свой таймаут для этого провайдера |

Порядок провайдеров и моделей в конфиге = порядок fallback-цепочки.
Конфиг перечитывается автоматически при изменении файла — можно добавлять ключи без рестарта.

## Переменные окружения (для Render / ClawCloud — ключи без правки файлов)

Env-переменные перекрывают config.json. В репозиторий кладётся config.json
с пустыми ключами, секреты задаются в панели (Render → Environment):

| Переменная | Что задаёт |
|------------|-----------|
| `ROUTER_API_KEY` | ключ доступа к самому роутеру (server.api_key) |
| `REQUIRE_AUTH` | `true` / `false` |
| `REQUEST_TIMEOUT` | таймаут запроса, сек |
| `PROVIDERS_JSON` | полный JSON-массив провайдеров (заменяет секцию providers) |
| `MISTRAL_API_KEY` | ключ Mistral (паттерн: `<ИМЯ>_API_KEY`, имя = name в верхнем регистре) |
| `MISTRAL_MODELS` | модели через запятую: `mistral-small-latest,ministral-8b-latest` |
| `MISTRAL_BASE_URL` | свой base_url |
| `OLLAMA_ENABLED` | `false` — выключить провайдера ollama (паттерн `<ИМЯ>_ENABLED`) |

Пример набора для Render: `ROUTER_API_KEY`, `MISTRAL_API_KEY`, `OLLAMA_ENABLED=false`.
Смена ключа = правка переменной в панели → сервис сам перезапускается, коммитить ничего не надо.

## Эндпоинты

| Метод | Путь | Ключ | Ответ |
|-------|------|------|-------|
| GET | `/` | нет | инфо о сервисе |
| GET | `/health` | нет | статус (для мониторинга/UptimeRobot) |
| GET | `/v1` | нет | `{"message": "v1"}` |
| GET | `/v1/models` | **да** | объединённый список моделей всех провайдеров |
| POST | `/v1/chat/completions` | **да** | чат, поддерживает `stream: true` (SSE) |
| POST | `/v1/embeddings` | **да** | эмбеддинги |
| GET | `/v1/providers` | **да** | статус/статистика провайдеров |

Без ключа на защищённых эндпоинтах — `401`.
Ключ передаётся как `Authorization: Bearer <api_key>` или заголовком `x-api-key`.

## Примеры

```bash
# ping
curl http://localhost:8000/v1

# список моделей (без ключа вернёт 401)
curl http://localhost:8000/v1/models -H "Authorization: Bearer CHANGE-ME-my-router-key-1"

# чат (модель можно не указывать — возьмёт первую из цепочки)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer CHANGE-ME-my-router-key-1" \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-small-latest", "messages": [{"role": "user", "content": "привет"}]}'
```

Клиент OpenAI (python):

```python
from openai import OpenAI

client = OpenAI(
    api_key="CHANGE-ME-my-router-key-1",   # ключ из config.json
    base_url="http://localhost:8000/v1",   # адрес роутера
)
resp = client.chat.completions.create(
    model="mistral-small-latest",          # или любое другое имя из /v1/models
    messages=[{"role": "user", "content": "привет"}],
)
print(resp.choices[0].message.content)
```

## Как работает роутинг

1. Клиент указал `model` — сначала пробуются все совпадения (формат `provider/model` или просто `model`),
   затем остальные модели цепочки.
2. Клиент НЕ указал `model` — обход всей цепочки по порядку конфига.
3. Переключение на следующую модель при: соединении не удалось, таймауте,
   HTTP 401, 403, 404, 408, 429, 5xx.
4. HTTP 400/422 — ошибка самого запроса, отдаётся клиенту без перебора.
5. Успешный ответ получает заголовки `x-router-provider`, `x-router-model`,
   `x-router-attempts` — видно, кто реально ответил.
6. Стриминг: fallback работает до первого отправленного байта.

## Деплой 24/7

### Вариант 1: ClawCloud Run (рекомендуется, без карты)

1. Залейте эти файлы в репозиторий на GitHub (не публичный, там ключи!).
2. clawcloudrun.com → Login with GitHub (аккаунт GitHub должен быть старше 180 дней
   — тогда каждый месяц бесплатно $5 кредитов, этого хватает на контейнер 24/7).
3. Create App → Import from GitHub (или соберите образ и укажите Docker image) →
   выберите репозиторий с Dockerfile, порт контейнера `8000`.
4. В Advanced/Environment добавьте переменную `PORT=8000`.
5. App откроется по HTTPS-URL вида `https://<name>.<region>.run.app` — это и есть ваш base_url.

### Вариант 2: бесплатный VPS (Hax / Woiden)

```bash
apt update && apt install -y python3-pip unzip
unzip llm-router.zip && cd llm-router
pip3 install -r requirements.txt
```

systemd-юнит `/etc/systemd/system/llm-router.service`:

```ini
[Unit]
Description=LLM Router
After=network.target

[Service]
WorkingDirectory=/opt/llm-router
ExecStart=/usr/bin/python3 server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now llm-router
```

Внимание: Hax/Woiden выдают IPv6-only VPS — для исходящих запросов к Mistral/Groq
нужен WARP (`warp-cli connect`) или NAT64. Продление VPS раз в ~5 дней в панели.

### Вариант 3: Render free

New → Web Service → репозиторий с Dockerfile → Instance Type: Free.
Минус: сервис засыпает после 15 минут без трафика. Лечится бесплатным пингером
(cron-job.org / UptimeRobot → GET `/health` каждые 10 минут).

### Ollama

Ollama — это локальный провайдер (`http://localhost:11434/v1`), он имеет смысл рядом
с роутером на вашей машине или на VPS с GPU. В бесплатном облаке Ollama не потянет —
там держите провайдера `ollama` с `"enabled": false`, либо пробросьте домашний Ollama
в облако через cloudflared tunnel и вставьте URL туннеля в `base_url`.

## Безопасность

- `config.json` содержит ключи провайдеров — не публикуйте его и не комитьте в публичный репозиторий.
- Обязательно поменяйте `api_key` в секции `server` на свой длинный случайный.
- Без ключа `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/providers` возвращают 401.
