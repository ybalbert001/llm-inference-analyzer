# LLM Inference Analyzer — Web Wrapper

对 `skills/llm-inference-analyzer` 的只读封装：列举/搜索已生成的分析报告，缺失时调用 skill 脚本生成并缓存（key = model ID + lang，固定默认参数）。不修改 skill 内任何代码。

## 本地运行

```bash
cd webapp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python import_existing.py     # 可选：导入 html_output/ 里已有的报告
DEV_MODE=1 .venv/bin/python app.py      # http://localhost:8000，免登录（身份=dev）
```

## 生产部署（HF OAuth）

1. 在 <https://huggingface.co/settings/applications> 创建 OAuth App：
   - Redirect URL: `https://<你的域名>/auth/callback`
   - Scopes: `openid profile`
2. 环境变量：

| 变量 | 说明 |
|---|---|
| `HF_OAUTH_CLIENT_ID` / `HF_OAUTH_CLIENT_SECRET` | OAuth App 凭证 |
| `BASE_URL` | 公网地址，如 `https://analyzer.example.com` |
| `SESSION_SECRET` | 随机串，cookie 签名（`openssl rand -hex 32`） |
| `ADMIN_USERS` | 可看 `/stats` 的 HF 用户名，逗号分隔（不设 = 任意登录用户可看） |
| `MAX_REPORTS` | 缓存报告上限，LRU 淘汰（默认 500） |
| `GENERATES_PER_HOUR` | 每用户每小时生成次数上限（默认 20） |

```bash
# 先填好这几个变量（建议存进 .env 或启动脚本，SESSION_SECRET 生成一次后固定，
# 否则每次重启所有用户都要重新登录）
export HF_OAUTH_CLIENT_ID="<OAuth App 的 client id>"
export HF_OAUTH_CLIENT_SECRET="<OAuth App 的 client secret>"
export BASE_URL="https://<域名>"              # 必须与 HF OAuth App 的 redirect URL 前缀一致
export SESSION_SECRET="$(openssl rand -hex 32)"  # 生成一次，之后固定复用

docker build -f webapp/Dockerfile -t llm-inference-analyzer-web .   # 在仓库根目录执行
docker run -d -p 8000:8000 -v analyzer-data:/app/webapp/data \
  -e HF_OAUTH_CLIENT_ID -e HF_OAUTH_CLIENT_SECRET \
  -e BASE_URL -e SESSION_SECRET \
  llm-inference-analyzer-web
```

`-e VAR`（不带 `=值`）表示把当前 shell 里同名环境变量原样传入容器，所以上面 export 之后无需再改 docker 命令。HF OAuth App 的 redirect URL 需登记为 `${BASE_URL}/auth/callback`（HF 支持登记多条，切域名时追加即可）。

当前生产环境的实例信息、SSM 登录方式、重新部署与运维手册见 `webapp/deploy.md`（含敏感信息，不入 git）。

## 设计要点

- **首页列表/搜索免登录；打开报告与生成均需 HF 登录**：未登录点开报告会经 OAuth 跳转后回到原报告页。OAuth 只取 username 做统计与限流，永不经手用户 token。
- **护栏**：model ID 正则校验；生成前 HEAD `config.json`（不存在/gated 直接拒绝，不进队列）；每用户限流；缓存 LRU 上限。服务器不设 `HF_TOKEN`，gated 模型不支持。
- **后台生成 + 轮询**：`POST /api/generate` → `GET /api/tasks/{id}`（2s 轮询）→ 完成跳转 `/reports/{slug}.html`。同一 model+lang 的并发请求合并到同一任务。
- **统计**：`/stats`（JSON）按用户/动作聚合 `access_log`（login/view/generate）。
- 强制重新生成：`POST /api/generate` 带 `"force": true`（skill 升级或模型 config 更新后用）。
