"""Method figure in a journal (Nature-style) layout: Graphviz diagrams + one matplotlib panel.

    python scripts/make_method_figure.py      # -> figures/method_figure.{svg,pdf,png}

a  agent-environment loop (Graphviz)
b  inside the environment, with our fatigue schedule (Graphviz)
c  what the body has vs what a training batch contains (matplotlib): because the 1M buffer
   never fills in a 500k-step run, a uniform batch's expected fatigue level is the running
   mean of l(t) so far, which lags the body's current level.
Needs: graphviz `dot`, `rsvg-convert`, matplotlib.
"""
import os
import re
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "figures"
SRC = os.path.join(OUT, "src")
os.makedirs(SRC, exist_ok=True)
FONT = "Liberation Sans"          # metric-compatible with Arial/Helvetica
AGENT, AGENT_F = "#178f65", "#eaf6f0"
ENV, ENV_F = "#2a6fbf", "#eaf1fa"
OURS, OURS_F = "#e0602a", "#fdeee5"
GREY, GREY_F, INK = "#6f6e69", "#f2f1ee", "#222222"

COMMON = f'''
  graph [fontname="{FONT}", fontsize=7, bgcolor="transparent", pad=0.02, margin=0];
  node  [fontname="{FONT}", fontsize=7, shape=box, style="rounded,filled", penwidth=0.6,
         margin="0.07,0.035", fontcolor="{INK}"];
  edge  [fontname="{FONT}", fontsize=6.5, penwidth=0.6, arrowsize=0.45, color="{GREY}",
         fontcolor="{INK}"];
'''


def it(v, sub=None):
    """Italic maths symbol with optional subscript, as a Graphviz HTML fragment."""
    return f"<I>{v}</I>" + (f"<SUB>{sub}</SUB>" if sub else "")


def card(title, color, lines, note=None):
    rows = [f'<TR><TD><B><FONT COLOR="{color}">{title}</FONT></B></TD></TR>']
    rows += [f"<TR><TD>{ln}</TD></TR>" for ln in lines]
    if note:
        rows.append(f'<TR><TD><FONT POINT-SIZE="6" COLOR="{GREY}">{note}</FONT></TD></TR>')
    return "<<TABLE BORDER=\"0\" CELLPADDING=\"0.6\" CELLSPACING=\"0\">" + "".join(rows) + "</TABLE>>"


PANEL_A = f'''digraph a {{ {COMMON}
  rankdir=TB; nodesep=0.55; ranksep=0.3; newrank=true;
  agent  [label={card("Agent (CrossQ)", AGENT,
                       ["actor " + it("π") + "(" + it("a") + "|" + it("s") + ")",
                        "critic " + it("Q") + "(" + it("s") + "," + it("a") + ")"],
                       "batch renorm.<BR/>no target network")},
          color="{AGENT}", fillcolor="{AGENT_F}"];
  env    [label={card("Environment", ENV, ["Hand Reach"], "see b")},
          color="{ENV}", fillcolor="{ENV_F}"];
  buffer [label={card("Replay buffer", GREY, ["1M transitions"], "holds the whole run")},
          color="{GREY}", fillcolor="{GREY_F}"];
  {{ rank=same; agent; env; }}
  agent -> env [label=<{it("a", "t")}: 39 muscle<BR/>activations>];
  env -> agent [label=<{it("s", "t")}>];
  env -> buffer [label=< transition<BR/> ({it("s", "t")}, {it("a", "t")}, {it("r", "t+1")}, {it("s", "t+1")} )>];
  buffer -> agent [label=<uniform  <BR/>batch of 256  >];
}}'''

PANEL_B = f'''digraph b {{ {COMMON}
  rankdir=TB; nodesep=0.12; ranksep=0.16;
  ain  [shape=plaintext, style="", label=<{it("a", "t")}>, height=0.15];
  rout [shape=plaintext, style="", label=<{it("r", "t+1")}>, height=0.15];
  sout [shape=plaintext, style="", label=<{it("s", "t+1")}>, height=0.15];
  subgraph cluster_env {{
    label=<<B><FONT COLOR="{ENV}">Environment</FONT></B>>; labeljust=l;
    style="rounded"; color="{ENV}"; penwidth=0.7; margin=7;
    fat [label={card("Fatigue schedule", OURS,
                     [it("ℓ") + "(" + it("t") + "),  " + it("t") + " = training step"],
                     "this work")},
         color="{OURS}", fillcolor="{OURS_F}", penwidth=0.9];
    dyn [label={card("Muscle dynamics", ENV,
                     ["MuJoCo, 39-muscle hand",
                      "MA + MR + MF = 1,  activation ≤ 1 − MF"])},
         color="{ENV}", fillcolor="white"];
    rew [label={card("Reward", ENV, ["−" + it("d", "t") + " + bonus − penalty"])},
         color="{ENV}", fillcolor="white"];
    st  [label={card("State", ENV, ["joints, fingertips,", "errors, activations"])},
         color="{ENV}", fillcolor="white"];
  }}
  fat -> dyn [label=<<FONT COLOR="{OURS}"> MF = {it("ℓ")}({it("t")})<BR/> at reset</FONT>>,
              color="{OURS}", penwidth=0.9];
  ain -> dyn;
  dyn -> rew; dyn -> st;
  rew -> rout; st -> sout;
}}'''


def dot(name, src):
    path = os.path.join(SRC, name + ".dot")
    open(path, "w").write(src)
    svg = os.path.join(SRC, name + ".svg")
    subprocess.run(["dot", "-Tsvg", path, "-o", svg], check=True)
    return svg


def panel_c(svg):
    plt.rcParams.update({"font.family": FONT, "font.size": 6, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "xtick.direction": "out", "ytick.direction": "out",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "path", "mathtext.fontset": "custom",
                         "mathtext.it": f"{FONT}:italic", "mathtext.rm": FONT})
    t = np.linspace(0.5, 500, 2000)                   # thousand steps
    drift = 0.5 * np.clip(t / 400, 0, 1)
    sudden = np.where(t >= 250, 0.5, 0.0)
    run_mean = lambda y: np.cumsum(y) / np.arange(1, len(y) + 1)  # noqa: E731  (uniform grid)
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.3), sharey=True)
    for ax, y, name, col in ((axes[0], drift, "Drift", "#eb6834"), (axes[1], sudden, "Sudden", "#eda100")):
        ax.plot(t, np.full_like(t, 0.5), color="#2a78d6", lw=0.7, alpha=0.7, zorder=1)
        ax.plot(t, y, color=col, lw=1.1, zorder=3)
        ax.plot(t, run_mean(y), color=col, lw=1.1, ls=(0, (2.2, 1.4)), zorder=3)
        ax.set_title(name, fontsize=6.5, weight="bold", pad=2, loc="left", color=INK)
        ax.set_xlim(0, 500); ax.set_ylim(-0.02, 0.56)
        ax.set_xticks([0, 250, 500]); ax.set_xticklabels(["0", "250k", "500k"])
        ax.set_yticks([0, 0.25, 0.5])
        ax.set_xlabel("training step", labelpad=1.5)
    axes[0].set_ylabel(r"fatigued fraction MF", labelpad=2)
    axes[0].text(500, 0.52, "Fixed (both lines)", ha="right", va="bottom", fontsize=5.8, color="#2a78d6")
    # direct labels on the Drift panel only (same meaning on both)
    axes[0].annotate(r"body, $\ell(t)$", xy=(280, 0.35), xytext=(140, 0.42), fontsize=5.8,
                     color=INK, ha="center", arrowprops=dict(arrowstyle="-", lw=0.4, color=INK))
    axes[0].annotate("training batch\n(buffer mean)", xy=(330, 0.206), xytext=(400, 0.06),
                     fontsize=5.8, color=INK, ha="center",
                     arrowprops=dict(arrowstyle="-", lw=0.4, color=INK))
    axes[1].annotate("still 0.25\nat 500k", xy=(495, 0.247), xytext=(430, 0.06), fontsize=5.8, color=INK,
                     ha="center", arrowprops=dict(arrowstyle="-", lw=0.4, color=INK))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.27, top=0.89, wspace=0.22)
    fig.savefig(svg, transparent=True)
    plt.close(fig)
    return svg


def size(svg_text):
    w = float(re.search(r'<svg[^>]*\swidth="([\d.]+)pt"', svg_text).group(1))
    h = float(re.search(r'<svg[^>]*\sheight="([\d.]+)pt"', svg_text).group(1))
    vb = re.search(r'viewBox="([^"]+)"', svg_text).group(1)
    return w, h, vb


def inner(svg_text):
    body = re.sub(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>|<!--.*?-->", "", svg_text, flags=re.S)
    return re.sub(r"^\s*<svg[^>]*>|</svg>\s*$", "", body.strip(), flags=re.S)


def compose(parts, out_svg, width=250.0, lead=11.0, gap=7.0):
    """Stack panels vertically; bold lower-case letters at top-left (journal style)."""
    y, chunks = 0.0, []
    for letter, path, center in parts:
        s = open(path).read()
        w, h, vb = size(s)
        k = min(1.0, (width - lead) / w)
        x = lead + ((width - lead - w * k) / 2 if center else 0)
        chunks.append(f'<text x="0" y="{y + 8:.1f}" font-family="{FONT}" font-weight="bold" '
                      f'font-size="9" fill="{INK}">{letter}</text>')
        chunks.append(f'<svg x="{x:.2f}" y="{y + 2:.2f}" width="{w * k:.2f}" height="{h * k:.2f}" '
                      f'viewBox="{vb}" xmlns="http://www.w3.org/2000/svg" '
                      f'xmlns:xlink="http://www.w3.org/1999/xlink">{inner(s)}</svg>')
        y += h * k + 2 + gap
    H = y - gap
    open(out_svg, "w").write(
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}pt" height="{H:.1f}pt" viewBox="0 0 {width} {H:.1f}">'
        f'<rect width="100%" height="100%" fill="white"/>' + "".join(chunks) + "</svg>")


if __name__ == "__main__":
    a = dot("method_a", PANEL_A)
    b = dot("method_b", PANEL_B)
    c = panel_c(os.path.join(SRC, "method_c.svg"))
    for p in (a, b, c):
        print(p, size(open(p).read())[:2])
    final = os.path.join(OUT, "method_figure.svg")
    compose([("a", a, True), ("b", b, True), ("c", c, False)], final)
    subprocess.run(["rsvg-convert", "-f", "pdf", final, "-o", final[:-4] + ".pdf"], check=True)
    subprocess.run(["rsvg-convert", "-z", "4", final, "-o", final[:-4] + ".png"], check=True)
    print("saved", final[:-4] + ".{svg,pdf,png}")
