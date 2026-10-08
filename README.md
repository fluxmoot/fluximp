# FluxImp

FluxImp is an AI agent harness with a desktop GUI that provides customizable, durable workflows for editing text files (i.e. code), doing web research via APIs from providers like [TinyFish](https://www.tinyfish.ai/) and [Exa](https://exa.ai/), running Python scripts, and running arbitrary test runners through Python plugins that normailze test failures.

FluxImp's approach to context management priotizes the direct conversation between yourself and the LLM by maintaining those messages verbatim.  Otherwise FluxImp is pretty aggressive about modifying the message history to limit context size and avoid context rot degrading LLM performance.

The GUI allows you to customize most of the system prompt the harness provides to the LLMs.  The only parts you can't customize through the GUI are the instructions for tool use that are partly programatically generated based on the available tools (i.e. the limitation technical & UX not philosphical).

![](docs\screenshot\Windows.png?raw=true)

## Quick Start

1) On Windows 10 or Windows 11 install Python 3.12 via [Miniforge Release 25.11.0-1](https://github.com/conda-forge/miniforge/releases/tag/25.11.0-1).  You could potentially use a newer version of Miniforge to install, but downgrading the Python version doesn't always work correctly.  Minforge 25.11.0-1 specifically installs version 3.12 of the Python.  If you already have Python 3.12 installed another way using that install should be fine.  **You can install Miniforge without it creating shortcuts or modifying the path.  In that case you need to use `<miniforge install dir>\python.exe` instead of `python` in the commands below.  Albeit slightly more hassle, this highly recommended to avoid system level headaches.**  You can create many independent, side-by-side Miniforge installations that use different versions of Python this way and they will not interfere with each other.

2) Install the FluxImp packages:

```console
python -m pip install fluximp-wxwidgets
```

3) Configure the Python keyring with at least one LLM inference provider API key:

```console
python -m keyring set deepinfra.com api_key
python -m keyring set friendli.ai api_key
```

4) [Optionally] configure the keyring with at least one web search & fetch provider API key:

```console
python -m keyring set tinyfish.ai api_key
python -m keyring set exa.ai api_key
python -m keyring set tavily.com api_key
python -m keyring set perplexity.ai api_key
```

5) Start the FluxImp desktop GUI

```console
python -m fluximp_wxwidgets --base-dir <e.g. %USERPROFILE%\fluximp or ~/fluximp>
```

6) Select a provider and model in the dropdowns at the bottom right (or use the defaults)

7) Enter a message in the edit box at the bottom center and press the Send button or SHIFT+Enter to start the workflow

## Clone and Run

Same steps as above but substitute the following for steps 2 and 5 to run it developer style.

2) Clone the repo and install the dependencies directly

```console
cd <repo parent dir>
git clone https://github.com/fluxmoot/fluximp.git
<miniforge install dir>\python.exe -m pip install openai==2.33.0 keyring==25.7.0 diff-match-patch==20241021 unidiff==0.7.5 python-ripgrep==0.0.9 langgraph==1.1.10 langchain-openai==1.2.1 rapidfuzz==3.14.3 wxPython==4.2.4 langgraph-checkpoint-sqlite==3.0.3 tavily-python==0.7.26 exa_py==2.16.0 linkup-sdk==0.18.3 parallel-web==1.1.0 perplexityai==0.40.0
```

5. Start the FluxImp desktop GUI

```console
cd <repo parent dir>
<miniforge install dir>\python.exe -m fluximp\wxwidgets_main.py --base-dir <e.g. %USERPROFILE%\fluximp or ~/fluximp>
```

## Workflows

You can customize all of the stock agents included in the stock Dev Workflow and you can create new workflows.   You can add new agents to the stock Dev Workflow and your custom workflows.  You can also define a sequence of custom agents (aka conditional agents) to be run when the LLM sends you a message with the `work_completion_summary` tag (e.g. a series of code review agents).  When a conditional agents runs and finds a problem its the text it provides in its report will be populate the message box, so you can send it as is, add to, edit it, or ignore it as you please.

FluxImp has a code-not-text philosphy due to LLMs frequently not fully following the rules and guidelines laid out in AGENTS.md / CLAUDE.md files.  As part of that philophy FluxImp does not currently directly support AGENTS.md (i.e. they are just treated as regular files with no special treatment).  Conditional agents are a replacement to the rules and guidelines that would go into AGENTS.md.  LLMs can only reason about limited number of things in one session, so a single review agent can't be relied on to catch violations of a large set of rules and guidlines.  Due to those limitations FluxImp supports running an arbitrary sequence of conditional agents rather than be limited to just one.

FluxImp workflows are durable in the sense that you can close the workflow tab or shutdown the application and then you can later reopen the workflow and continue from right where you left off.  The workflow will likely be durable in the face of either the GUI or the agent process crashing or being killed (this has been excercised many times while dogfooding; there's always non-recoverable edge cases so it isn't 100% thus the 'likely' caveat).

FluxImp workflows are also durable in the sense that you can go back to an earlier message you sent and fork it into a new message (or just repeat the same message to try your luck again).  When you fork a workflow the harness managed files will be rolled back to their contents at that point in the history (this is whole file restoration; its not a merge and will stomp any manual edits).  The harness maintains this state itself indepedently of any version control system (i.e. git is not required and is not iteracted with if present).

Forks are not reversible.  You can not go back to the branch before a fork.

## Context Mangement

**The primary goal of FluxImp's context management strategy is to maximize the amount of quality work done in each workflow.**  *The secondary goal is to maximize cached tokens in billing metrics.*

Compaction eating the direct messages between yourself and the LLM is more likely to halt quality work than not, so compaction is not done.  Your messages to the LLM and the LLM's messages to you will always be included verbatim in the API requests.

FluxImp does rolling summarization of the intermediate messages, which have most of the reasoning and tool uses, that come in between your message and the LLMs direct response message to you.  The rolling summaries are then included verbatim.  Summaries are not ruther summarized on the theory that this avoids information loss that would impair work quality.  It would be interesting to test this thoery, but a benchmark of sufficient length and complexity to test this not in hand yet.

FluxImp provides a set of always fresh file and query views that are collectively limited to a fixed byte size (i.e. one or more are automatically closed when the size of all views is too big).  FluxImp also the includes the current, cumulative diff of the file edits on each inference API request.  The thinking here is to avoid having (potentially numerous) older versions of file contents in the context window and allow the LLMs to avoid generating tool uses to reread files they already have in their view on.  Also the diff allows LLMs see what edits have been made without needing to compare different versions of the content in their reasoning.

While the context is very actively managed and this necessitates breaking the prefix cache, the implementation tries to do this as effieciently as possible to balance both the cost and time advantages of prefix caching while also reducing the total prompt tokens per request to help avoid context rot.

A query view is a specific ripgrep search or a specific find files search.  A file view is potentially the entire file or one more windows into the file.  The content of the views provided for each inference API request is always current for the filesystem state (i.e. ripgrep views are rerun before every request; this can be slow for broad searches at the root of large workspaces).

## Tools for LLMs

FluxImp does not provide LLMs with tools to use shells such as bash, powershell, or cmd.  Instead FluxImp provides LLMs with specialized tools for ripgrep, finding files, creating & deleting files, efficiently editing text files, running a Python script by path (not arbitrary code in to the tool use).

The most common way LLMs corrupt text files is by rewriting them in their entirety.  There is actually a tool for writing files implmented in FluxImp's code, but its been disabled.  The tool for creating files will undo the deletion of the same file as LLMs just really, really want to rewrite files tried to get around the lack of a write file tool by deleting the file and recreating.  This harness forces them to edit files surgically to prevent wholesale file corruption.

FluxImp also provides LLMs with tools for web search and web fetch that do not use the local browser and instead use purpose built provider APIs.  You'll need an account with the web search/fetch API providers to get an API key just like with the LLM inference API providers.  Its been really interesting using a research agent with these tools, as they come up with results that don't show up in the big search engines.  It really shows how enshitified the big search engines are.

Finally FluxImp can provide LLMs with a run_tests tool if you integrate your own Python plugins to normalize the output of your test runners.  The test runners themselves don't need to written in Python, just the plugins.  This approach is arises from the code-not-text philosphy.  Instead of bunch of text to inform LLMs on who to run the tests and and navigate diagnostics just create code that does it deterministically instead instead of burning tokesn making LLMs run throught the same reasoning repeatedly (or worse yet burning tokens scanning verbose test outputs for to determine there are even failiures or not).

## License

MPL-2.0 — see [LICENSE](http://mozilla.org/MPL/2.0/).
