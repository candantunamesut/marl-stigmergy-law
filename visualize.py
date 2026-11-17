import os
import sys
import json
import argparse
import numpy as np
import torch

try:
    from env import MARLSurvivalEnv, ACTION_NAMES
    from model import ActorCritic
    from config import CONFIG
except ImportError:
    from marl_survival_experiment.env import MARLSurvivalEnv, ACTION_NAMES
    from marl_survival_experiment.model import ActorCritic
    from marl_survival_experiment.config import CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "marl_survival_model.pt")
DEFAULT_HTML_PATH = os.path.join(BASE_DIR, "simulation_playback.html")


def generate_html_playback(frames_data: list, output_path: str = DEFAULT_HTML_PATH, grid_size: int = 20):
    """
    Generates a standalone, rich interactive HTML visualization player
    to view simulation steps with play/pause, step slider, health bars, and event logs.
    """
    frames_json = json.dumps(frames_data)
    html_content = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MARL Kollektif Hayatta Kalma Simulasyonu</title>
    <style>
        :root {{
            --bg-dark: #0f172a;
            --panel-bg: #1e293b;
            --border-color: #334155;
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --text-light: #f8fafc;
            --text-muted: #94a3b8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-dark);
            color: var(--text-light);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            margin-bottom: 1.5rem;
        }}
        .header h1 {{
            font-size: 1.8rem;
            color: var(--accent-blue);
            margin-bottom: 0.3rem;
        }}
        .header p {{
            color: var(--text-muted);
            font-size: 0.95rem;
        }}
        .container {{
            display: flex;
            gap: 2rem;
            max-width: 1200px;
            width: 100%;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .grid-panel {{
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        #grid {{
            display: grid;
            grid-template-columns: repeat({grid_size}, 22px);
            grid-template-rows: repeat({grid_size}, 22px);
            gap: 2px;
            background: #090d16;
            padding: 6px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .cell {{
            width: 22px;
            height: 22px;
            background: #151f32;
            border-radius: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: bold;
            user-select: none;
            transition: all 0.15s ease;
        }}
        .cell.food {{
            background: #064e3b;
            color: #34d399;
            box-shadow: 0 0 6px rgba(52, 211, 153, 0.4);
        }}
        .cell.agent-0 {{ background: #1d4ed8; color: #fff; box-shadow: 0 0 8px rgba(59, 130, 246, 0.6); }}
        .cell.agent-1 {{ background: #b91c1c; color: #fff; box-shadow: 0 0 8px rgba(239, 68, 68, 0.6); }}
        .cell.agent-2 {{ background: #b45309; color: #fff; box-shadow: 0 0 8px rgba(245, 158, 11, 0.6); }}
        .cell.agent-3 {{ background: #6d28d9; color: #fff; box-shadow: 0 0 8px rgba(139, 92, 246, 0.6); }}

        .sidebar {{
            flex: 1;
            min-width: 320px;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        .card {{
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.2rem;
        }}
        .card-title {{
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.4rem;
        }}
        .agent-row {{
            margin-bottom: 0.8rem;
        }}
        .agent-header {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
        }}
        .bar-container {{
            background: #090d16;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            transition: width 0.2s ease, background-color 0.2s ease;
        }}
        .controls {{
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }}
        .btn-row {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }}
        button {{
            background: #2563eb;
            color: #fff;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
        }}
        button:hover {{ background: #1d4ed8; }}
        button:disabled {{ background: #475569; cursor: not-allowed; }}
        input[type="range"] {{
            width: 100%;
            accent-color: var(--accent-blue);
            cursor: pointer;
        }}
        .log-box {{
            max-height: 140px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.4;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }}
        .log-item {{
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            background: #151f32;
        }}
        .log-food {{ color: #34d399; }}
        .log-death {{ color: #f87171; }}
        .log-shared {{ color: #fbbf24; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MARL Kollektif Hayatta Kalma & Kaynak Paylaşımı Simulasyonu</h1>
        <p>20x20 Izgara Alanında Bağımsız Ajanların Hayatta Kalma Dinamikleri ve Takım Ödülü</p>
    </div>

    <div class="container">
        <div class="grid-panel">
            <div id="grid"></div>
        </div>

        <div class="sidebar">
            <div class="card controls">
                <div class="card-title">
                    <span>Oynatma Kontrolleri</span>
                    <span id="step-label" style="color: var(--accent-blue); font-size: 0.9rem;">Adım: 0 / 0</span>
                </div>
                <input type="range" id="step-slider" min="0" max="0" value="0">
                <div class="btn-row">
                    <button id="btn-play">Oynat</button>
                    <button id="btn-prev">Önceki</button>
                    <button id="btn-next">Sonraki</button>
                    <button id="btn-reset">Başa Dön</button>
                    <select id="speed-select" style="background:#090d16; color:#fff; border:1px solid var(--border-color); padding:0.4rem; border-radius:6px; margin-left:auto;">
                        <option value="400">0.5x</option>
                        <option value="200" selected>1.0x</option>
                        <option value="100">2.0x</option>
                        <option value="40">5.0x</option>
                    </select>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Ajan Sağlık Durumları (Maks 60 HP)</div>
                <div id="agents-panel"></div>
            </div>

            <div class="card">
                <div class="card-title">Adım Olayları ve Ödüller</div>
                <div class="log-box" id="event-logs"></div>
            </div>
        </div>
    </div>

    <script>
        const frames = {frames_json};
        const GRID_SIZE = {grid_size};
        let currentFrameIndex = 0;
        let isPlaying = false;
        let playTimer = null;

        const gridEl = document.getElementById("grid");
        const slider = document.getElementById("step-slider");
        const stepLabel = document.getElementById("step-label");
        const btnPlay = document.getElementById("btn-play");
        const btnPrev = document.getElementById("btn-prev");
        const btnNext = document.getElementById("btn-next");
        const btnReset = document.getElementById("btn-reset");
        const speedSelect = document.getElementById("speed-select");
        const agentsPanel = document.getElementById("agents-panel");
        const eventLogs = document.getElementById("event-logs");

        slider.max = Math.max(0, frames.length - 1);

        // Grid cells init
        const cells = [];
        for (let y = 0; y < GRID_SIZE; y++) {{
            for (let x = 0; x < GRID_SIZE; x++) {{
                const cell = document.createElement("div");
                cell.className = "cell";
                gridEl.appendChild(cell);
                cells.push(cell);
            }}
        }}

        function renderFrame(idx) {{
            if (!frames[idx]) return;
            const f = frames[idx];
            currentFrameIndex = idx;
            slider.value = idx;
            stepLabel.textContent = `Adım: ${{f.step}} / ${{frames[frames.length - 1].step}}`;

            // Reset grid
            cells.forEach(c => {{
                c.className = "cell";
                c.textContent = "";
            }});

            // Place food
            f.foods.forEach(food => {{
                if (food.active) {{
                    const cIdx = food.y * GRID_SIZE + food.x;
                    if (cells[cIdx]) {{
                        cells[cIdx].classList.add("food");
                        cells[cIdx].textContent = "F";
                    }}
                }}
            }});

            // Place agents
            f.agents.forEach(a => {{
                if (a.alive) {{
                    const cIdx = a.y * GRID_SIZE + a.x;
                    if (cells[cIdx]) {{
                        cells[cIdx].classList.add(`agent-${{a.id}}`);
                        cells[cIdx].textContent = a.id;
                    }}
                }}
            }});

            // Render health bars
            agentsPanel.innerHTML = "";
            f.agents.forEach(a => {{
                const pct = Math.max(0, Math.min(100, (a.health / 60.0) * 100));
                let color = "#22c55e";
                if (pct < 30) color = "#ef4444";
                else if (pct < 60) color = "#f59e0b";

                const row = document.createElement("div");
                row.className = "agent-row";
                row.innerHTML = `
                    <div class="agent-header">
                        <span><strong>Ajan ${{a.id}}</strong> ${{a.alive ? "" : "(ÖLDÜ)"}}</span>
                        <span>${{a.health.toFixed(1)}} / 60 HP</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${{pct}}%; background: ${{color}};"></div>
                    </div>
                `;
                agentsPanel.appendChild(row);
            }});

            // Render logs
            eventLogs.innerHTML = "";
            if (f.events && f.events.length > 0) {{
                f.events.forEach(ev => {{
                    const item = document.createElement("div");
                    item.className = "log-item";
                    if (ev.includes("Yemek")) item.classList.add("log-food");
                    else if (ev.includes("Öldü")) item.classList.add("log-death");
                    else if (ev.includes("%10")) item.classList.add("log-shared");
                    item.textContent = ev;
                    eventLogs.appendChild(item);
                }});
            }} else {{
                eventLogs.innerHTML = '<div style="color:#64748b;">Bu adımda özel bir olay gerçekleşmedi.</div>';
            }}
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            btnPlay.textContent = isPlaying ? "Durdur" : "Oynat";
            if (isPlaying) {{
                runPlay();
            }} else {{
                clearTimeout(playTimer);
            }}
        }}

        function runPlay() {{
            if (!isPlaying) return;
            if (currentFrameIndex < frames.length - 1) {{
                renderFrame(currentFrameIndex + 1);
                playTimer = setTimeout(runPlay, parseInt(speedSelect.value));
            }} else {{
                isPlaying = false;
                btnPlay.textContent = "Oynat";
            }}
        }}

        btnPlay.addEventListener("click", togglePlay);
        btnPrev.addEventListener("click", () => {{
            if (currentFrameIndex > 0) renderFrame(currentFrameIndex - 1);
        }});
        btnNext.addEventListener("click", () => {{
            if (currentFrameIndex < frames.length - 1) renderFrame(currentFrameIndex + 1);
        }});
        btnReset.addEventListener("click", () => {{
            renderFrame(0);
        }});
        slider.addEventListener("input", (e) => {{
            renderFrame(parseInt(e.target.value));
        }});

        // Initial render
        renderFrame(0);
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    abs_url = "file://" + os.path.abspath(output_path)
    print(f"\n[OK] Interactive HTML Simulation visualizer exported to: {output_path}")
    print(f"     Opening in browser: {abs_url}")
    try:
        import webbrowser
        webbrowser.open(abs_url)
    except Exception:
        pass


def visualize_episode(model_path: str = None, max_steps: int = 80, seed: int = 42, export_html: bool = True, visibility_mode: str = "local"):
    print("=" * 65)
    print(f"RUNNING VISUAL SIMULATION OF MARL STIGMERGY AGENTS ({visibility_mode.upper()} VISIBILITY)")
    print("=" * 65)

    cfg = CONFIG.copy()
    cfg["VISIBILITY_MODE"] = visibility_mode
    env = MARLSurvivalEnv(config=cfg, max_steps=max_steps)
    model = ActorCritic(obs_dim=30, action_dim=6, hidden_dim=128)

    if model_path is None:
        cand_paths = [
            os.path.join(BASE_DIR, f"marl_survival_model_{visibility_mode}.pt"),
            os.path.join(BASE_DIR, "marl_survival_model.pt"),
            os.path.join(BASE_DIR, "evolution_best_model.pt"),
        ]
        for p in cand_paths:
            if os.path.exists(p):
                model_path = p
                break

    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded trained policy weights from {model_path}")
        except Exception as e:
            print(f"Notice: Could not load trained weights ({e}). Running baseline policy.")
    else:
        print(f"Notice: No checkpoint found. Running with random/untrained policy.")
    model.eval()

    obs_dict, _ = env.reset(seed=seed)
    frames_data = []

    # Record initial frame (step 0)
    init_agents_data = []
    for a in env.possible_agents:
        pos = env.agent_positions[a]
        init_agents_data.append({
            "id": int(a.split("_")[-1]),
            "name": a,
            "x": int(pos[0]),
            "y": int(pos[1]),
            "health": float(env.agent_healths[a]),
            "alive": True,
        })
    init_foods_data = [
        {"x": int(r["pos"][0]), "y": int(r["pos"][1]), "active": bool(r["active"])}
        for r in env.rewards_data
    ]
    frames_data.append({
        "step": 0,
        "agents": init_agents_data,
        "foods": init_foods_data,
        "events": ["Simülasyon başlangıcı. Tüm ajanlar 60 HP ile yerleşti."],
    })

    for step in range(1, max_steps + 1):
        if not env.agents:
            print(f"All agents died at step {step - 1}.")
            break

        actions = {}
        with torch.no_grad():
            for agent in env.agents:
                obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32).unsqueeze(0)
                logits, _ = model(obs_t)
                action = torch.argmax(logits, dim=-1).item()
                actions[agent] = action

        # Print current terminal grid
        print(f"\n--- STEP {step} ---")
        grid = [["." for _ in range(env.grid_size)] for _ in range(env.grid_size)]

        # Place pheromone marks (~ for faint, * for strong)
        for y in range(env.grid_size):
            for x in range(env.grid_size):
                val = env.shared_map[y, x]
                if val > 0.5:
                    grid[y][x] = "*"
                elif val > 0.1:
                    grid[y][x] = "~"

        # Place food pools
        for r in env.rewards_data:
            if r["amount"] > 0:
                rx, ry = r["pos"]
                grid[ry][rx] = "F"

        for agent in env.agents:
            ax, ay = env.agent_positions[agent]
            agent_id = agent.split("_")[-1]
            grid[ay][ax] = agent_id

        for row in grid:
            print(" ".join(row))

        hp_str = ", ".join([f"{a}: {env.agent_healths[a]:.1f}HP" for a in env.agents])
        act_str = ", ".join([f"{a}: {ACTION_NAMES[actions[a]]}" for a in env.agents])
        pool_str = ", ".join([f"P{i}:{r['amount']:.1f}/{r['capacity']:.0f}" for i, r in enumerate(env.rewards_data)])
        print(f"Healths: [{hp_str}]")
        print(f"Actions: [{act_str}]")
        print(f"Pools:   [{pool_str}] | Marked Cells: {np.count_nonzero(env.shared_map > 0.05)}")

        step_events = []
        obs_dict, rewards, terminations, truncations, infos = env.step(actions)

        for agent, r in rewards.items():
            if infos.get(agent, {}).get("wrote_to_map"):
                ev = f"*** {agent} HARITAYA FEROMON IZI BIRAKTI (Stigmergy Mark)!"
                print(f"  {ev}")
                step_events.append(ev)
            if infos.get(agent, {}).get("food_found"):
                consumed = infos[agent].get("food_consumed", 0.0)
                ev = f">>> {agent} YEMEK YEDI (+{consumed:.1f} birim beslendi)!"
                print(f"  {ev}")
                step_events.append(ev)
            if "shared_food_reward" in infos.get(agent, {}):
                ev = f">>> {agent} takim arkadasi yemeginden pay aldi (+{infos[agent]['shared_food_reward']:.2f} Odul)!"
                print(f"  {ev}")
                step_events.append(ev)
            if infos.get(agent, {}).get("dead"):
                ev = f"!!! {agent} OLDU (-100 Odul). Diger tum ajanlar -10 HP kaybetti!"
                print(f"  {ev}")
                step_events.append(ev)

        # Record frame for HTML player
        agents_frame = []
        for a in env.possible_agents:
            pos = env.agent_positions[a]
            is_alive = a in env.agents
            agents_frame.append({
                "id": int(a.split("_")[-1]),
                "name": a,
                "x": int(pos[0]),
                "y": int(pos[1]),
                "health": float(env.agent_healths[a]),
                "alive": is_alive,
            })
        foods_frame = [
            {"x": int(r["pos"][0]), "y": int(r["pos"][1]), "active": bool(r["active"])}
            for r in env.rewards_data
        ]
        frames_data.append({
            "step": step,
            "agents": agents_frame,
            "foods": foods_frame,
            "events": step_events,
        })

    if export_html:
        generate_html_playback(frames_data, DEFAULT_HTML_PATH, env.grid_size)


def main():
    parser = argparse.ArgumentParser(description="Visualize MARL Survival Agents")
    parser.add_argument("--steps", type=int, default=80, help="Maximum number of simulation steps to run")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--visibility", type=str, default="local", choices=["local", "full"], help="Visibility configuration: 'local' or 'full'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment")
    parser.add_argument("--no-html", action="store_true", help="Disable HTML playback export")
    args = parser.parse_args()

    visualize_episode(
        model_path=args.model,
        max_steps=args.steps,
        seed=args.seed,
        export_html=not args.no_html,
        visibility_mode=args.visibility,
    )


if __name__ == "__main__":
    main()
