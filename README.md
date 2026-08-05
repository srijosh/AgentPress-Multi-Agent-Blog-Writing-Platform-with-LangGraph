# 📝 AgentPress — Multi Agent Blog Writing Platform with LangGraph

A technical blog generation app that transforms a single topic prompt into a structured, research-aware Markdown article. The system uses a multi-agent orchestration workflow built with LangGraph, FastAPI, and local Ollama-powered prompting to stream live execution steps to a clean web interface.

It is designed as a lightweight multi-agent writing pipeline:

- route the topic into a research or closed-book flow
- plan a blog outline with concrete tasks
- generate sections in parallel
- merge the output into a final markdown article
- optionally generate supporting images for the article

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Tools and Technologies](#tools-and-technologies)

## Introduction

Writing high-quality technical content manually requires hours of background research, structured outlining, drafting separate components, and formatting markdown files. This project brings that complete editorial workflow into an automated, single-click application. By modeling the writing process as an advanced multi-agent pipeline, the system breaks down complex technical topics, outlines specific chapter goals, writes sections in parallel, and renders a production-ready article.

## Features

- Multi-step LangGraph workflow for blog planning and writing
- FastAPI + Jinja frontend for an interactive web experience
- Optional research mode using Tavily web search
- Markdown article generation with section-by-section writers
- Image placeholder planning and Gemini image generation support

## Installation

1. Clone the repository:

```bash
git clone https://github.com/srijosh/AgentPress-Multi-Agent-Blog-Writing-Platform-with-LangGraph.git
```

2. Navigate to the project directory:

```bash
cd AgentPress-Multi-Agent-Blog-Writing-Platform-with-LangGraph
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file from `.env.sample`.

5. Start Ollama and pull the model

```bash
ollama list
ollama pull llama3.1:8b
```

## Usage

Start the FastAPI application server locally:

```bash
python app.py
```

Once running, open your web browser and navigate to:

```text
http://127.0.0.1:8000/
```

## Tools and Technologies

- **LangGraph & LangChain**: Manages the multi-agent state, worker nodes, and parallel execution logic.
- **FastAPI**: Asynchronous web framework used for backend routes and streaming live execution data.
- **Ollama**: Local model engine used to serve open-source LLMs like Llama 3.1 privately.
- **Jinja2Templates**: Server-side template engine used to load and render the HTML dashboard.
- **HTML / CSS / JavaScript**: Frontend stack used to handle the UI, read SSE streams, and power the stop button.
- **Tavily API**: Web search engine used by the research node to gather live content and facts.
- **Google GenAI (Gemini)**: Multimodal AI engine used to generate the supporting images for articles.
