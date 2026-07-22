# =============================================================================
# MindGuard AI – Mental Health Awareness & Suicide Prevention Agent
# =============================================================================
# Built with: Python, Flask, IBM watsonx.ai Studio, IBM Granite Models, RAG
# Architecture: Agentic AI with Multi-Agent Orchestration
# Purpose: Mental Health Awareness, Emotional Support, Suicide Prevention
#
# Agents:
#   1. Mental Health Awareness Agent
#   2. Emotional Support Agent
#   3. Distress Detection Agent
#   4. Prevention & Wellness Agent
#   5. Human Support Connector Agent
#   6. Master Orchestrator Agent
#
# Suitable for: IBM SkillsBuild, Hackathons, Academic Projects, AI Showcases
# =============================================================================

import os
import re
import io
import json
import math
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

# Load .env file automatically (python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to system env vars

# PDF text extraction (optional — gracefully skipped if not installed)
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# IBM watsonx.ai SDK
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    WATSONX_SDK_AVAILABLE = True
except ImportError:
    WATSONX_SDK_AVAILABLE = False

# =============================================================================
# Flask Application Initialization
# =============================================================================
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MindGuardAI")

# =============================================================================
# IBM watsonx.ai Credentials (read from environment variables)
# Set these before running:
#   export WATSONX_API_KEY="your-api-key"
#   export WATSONX_PROJECT_ID="your-project-id"
#   export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
# =============================================================================
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# IBM Granite Model ID — primary reasoning engine
GRANITE_MODEL_ID = "ibm/granite-13b-instruct-v2"

# =============================================================================
# IBM watsonx.ai Model Client Initialization
# =============================================================================
watsonx_model = None

def initialize_watsonx():
    """
    Initialize the IBM watsonx.ai ModelInference client using Granite.
    Falls back gracefully to demo mode if credentials are missing.
    """
    global watsonx_model
    if not WATSONX_SDK_AVAILABLE:
        logger.warning("ibm-watsonx-ai SDK not installed. Running in demo mode.")
        return False
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        logger.warning("IBM watsonx.ai credentials not set. Running in demo mode.")
        return False
    try:
        credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        watsonx_model = ModelInference(
            model_id=GRANITE_MODEL_ID,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={
                GenParams.MAX_NEW_TOKENS: 512,
                GenParams.TEMPERATURE: 0.7,
                GenParams.TOP_P: 0.9,
                GenParams.REPETITION_PENALTY: 1.1,
            }
        )
        logger.info("IBM watsonx.ai Granite model initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"watsonx.ai initialization failed: {e}")
        return False

WATSONX_ACTIVE = initialize_watsonx()

# =============================================================================
# Core IBM watsonx.ai Generation Function
# =============================================================================
def generate_response(prompt: str, max_tokens: int = 512) -> str:
    """
    Send a prompt to IBM Granite via watsonx.ai and return the generated text.
    Falls back to intelligent demo responses when credentials are unavailable.

    Args:
        prompt (str): The instruction/question prompt
        max_tokens (int): Maximum tokens to generate

    Returns:
        str: Generated response from IBM Granite
    """
    # --- IBM watsonx.ai Granite Call ---
    if WATSONX_ACTIVE and watsonx_model:
        try:
            response = watsonx_model.generate_text(prompt=prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Granite generation error: {e}")
            return _demo_response(prompt)
    return _demo_response(prompt)


def _demo_response(prompt: str) -> str:
    """
    Intelligent fallback demo responses when IBM watsonx.ai is not connected.
    Demonstrates what the Granite model would return.
    """
    p = prompt.lower()
    if "anxiety" in p:
        return ("Anxiety is a natural emotional response to perceived threats or uncertainty. "
                "It becomes a concern when it's persistent and interferes with daily life. "
                "Common symptoms include racing thoughts, increased heart rate, sweating, and a sense of dread. "
                "Evidence-based treatments include Cognitive Behavioral Therapy (CBT), mindfulness practices, "
                "and in some cases, medication prescribed by a healthcare professional.")
    if "depression" in p:
        return ("Depression is more than feeling sad — it is a serious mental health condition characterized by "
                "persistent low mood, loss of interest, fatigue, changes in sleep and appetite, and difficulty "
                "concentrating. It affects millions worldwide. Effective treatments include therapy, lifestyle changes, "
                "social support, and professional medical care when needed.")
    if "burnout" in p:
        return ("Burnout is a state of chronic stress leading to physical and emotional exhaustion, cynicism, and "
                "feelings of ineffectiveness. It is common in high-demand work environments. Signs include constant "
                "fatigue, reduced performance, detachment, and irritability. Recovery involves rest, boundary-setting, "
                "social connection, and sometimes professional counseling.")
    if "mindfulness" in p:
        return ("Mindfulness is the practice of intentionally focusing attention on the present moment without judgment. "
                "Research shows it reduces stress, anxiety, and depression while improving emotional regulation. "
                "Simple practices include mindful breathing, body scans, and mindful observation of thoughts without attachment.")
    if "stress" in p:
        return ("Stress is the body's response to challenging demands. Healthy stress (eustress) motivates us, while "
                "chronic stress harms physical and mental health. Effective management strategies include regular exercise, "
                "deep breathing, time management, social support, limiting screen time, and adequate sleep.")
    if "overwhelm" in p or "lonely" in p or "alone" in p or "nobody" in p:
        return ("I hear you, and I want you to know that what you're feeling is completely valid. "
                "Feeling overwhelmed or lonely can be incredibly difficult, and it takes courage to acknowledge it. "
                "You are not alone — many people experience these feelings. It's okay to take things one step at a time. "
                "Would it help to talk about what's been weighing on you?")
    if "risk" in p or "distress" in p or "hopeless" in p:
        return ("Based on the emotional signals detected, there are indicators of elevated distress. "
                "It is strongly recommended to reach out to a trusted person or mental health professional. "
                "You deserve support, and help is available. Please consider contacting a crisis helpline if you feel unsafe.")
    if "wellness" in p or "coping" in p or "exercise" in p:
        return ("A personalized wellness plan for you: Start each morning with 5 minutes of deep breathing. "
                "Take a 20-minute walk outdoors daily. Write 3 things you are grateful for each evening. "
                "Limit screen time to 2 hours after work. Practice progressive muscle relaxation before bed. "
                "Reach out to one trusted friend or family member this week.")
    if "help" in p or "support" in p or "resource" in p:
        return ("There are many compassionate professionals and resources ready to help. "
                "The WHO recommends speaking with a licensed therapist or counselor as a first step. "
                "Community mental health centers, online therapy platforms, and peer support groups are all valuable options. "
                "Remember: seeking help is a sign of strength, not weakness.")
    return ("Thank you for sharing. MindGuard AI is here to support your mental health journey. "
            "Whether you need information, coping strategies, or just someone to listen, I'm here. "
            "Remember, your well-being matters and professional help is always available.")


# =============================================================================
# LIGHTWEIGHT RAG SYSTEM
# =============================================================================
# Simple in-memory vector store using TF-IDF style cosine similarity.
# No external vector database required.
# =============================================================================

rag_documents = []   # Stores chunked document passages
rag_metadata   = []  # Stores source metadata per chunk


def _tokenize(text: str) -> dict:
    """Convert text to a simple term-frequency dictionary."""
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    tf = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    return tf


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two TF dictionaries."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[w] * vec_b[w] for w in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def ingest_document(text: str, source: str = "uploaded_document"):
    """
    Ingest a document into the RAG knowledge base.
    Splits text into ~300-word chunks and stores TF vectors.

    Args:
        text (str): Raw document text
        source (str): Document source label
    """
    words = text.split()
    chunk_size = 300
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    for chunk in chunks:
        if len(chunk.strip()) > 50:
            rag_documents.append({"text": chunk, "vector": _tokenize(chunk)})
            rag_metadata.append({"source": source, "chars": len(chunk)})
    logger.info(f"RAG: Ingested {len(chunks)} chunks from '{source}'.")


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve the most relevant passages from the RAG knowledge base.
    Uses cosine similarity between query and stored document chunks.

    Args:
        query (str): User query or agent prompt
        top_k (int): Number of top passages to return

    Returns:
        str: Concatenated relevant passages, or empty string if no documents
    """
    if not rag_documents:
        return ""
    query_vec = _tokenize(query)
    scored = [(i, _cosine_similarity(query_vec, doc["vector"])) for i, doc in enumerate(rag_documents)]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]
    passages = []
    for idx, score in top:
        if score > 0.05:
            src = rag_metadata[idx]["source"]
            passages.append(f"[Source: {src}]\n{rag_documents[idx]['text']}")
    return "\n\n---\n\n".join(passages)


# Pre-load built-in mental health knowledge into RAG
BUILTIN_KNOWLEDGE = """
Mental health is a state of mental well-being that enables people to cope with the stresses of life,
realize their abilities, learn well and work well, and contribute to their community.
Mental health is a basic human right. And it is crucial to personal, community and socio-economic development.
Mental health is more than the absence of mental disorders. It exists on a complex continuum.

Anxiety disorders are the most common mental health disorders globally. They include generalized anxiety disorder,
panic disorder, social anxiety disorder, and specific phobias. Symptoms include excessive worry, restlessness,
fatigue, difficulty concentrating, irritability, muscle tension, and sleep disturbance.

Depression is a leading cause of disability worldwide. Symptoms include persistent sadness, loss of interest,
changes in appetite and sleep, fatigue, feelings of worthlessness or guilt, difficulty thinking, and in severe
cases, thoughts of death or suicide. Treatment includes psychotherapy, antidepressants, and lifestyle changes.

Suicide prevention requires a comprehensive multi-level approach. Warning signs include talking about wanting
to die, looking for ways to kill oneself, talking about being a burden, increasing substance use, withdrawing
from activities, extreme mood swings, and expressing feelings of hopelessness. If you notice these signs,
take them seriously and encourage the person to seek help immediately.

Crisis resources: National Suicide Prevention Lifeline: 988 (US). Crisis Text Line: Text HOME to 741741.
International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
Befrienders Worldwide: https://www.befrienders.org/

Mindfulness-Based Stress Reduction (MBSR) is an evidence-based program that helps people manage stress,
anxiety, depression and pain through mindfulness meditation and yoga. Studies show it significantly reduces
psychological distress and improves quality of life.

Cognitive Behavioral Therapy (CBT) is the gold standard treatment for many mental health conditions.
It works by identifying and changing negative thought patterns and behaviors. CBT has strong evidence
for treating depression, anxiety, PTSD, OCD, eating disorders, and substance abuse.

Self-care strategies for mental wellness include regular physical exercise (150 minutes/week),
maintaining a consistent sleep schedule (7-9 hours), eating a balanced diet, limiting alcohol and caffeine,
practicing relaxation techniques, maintaining social connections, and engaging in meaningful activities.

Burnout is characterized by emotional exhaustion, depersonalization, and reduced sense of personal
accomplishment. It results from prolonged chronic workplace stress. Recovery requires rest, boundary-setting,
seeking support, and sometimes changing the work environment.
"""
ingest_document(BUILTIN_KNOWLEDGE, source="WHO Mental Health Guidelines (Built-in)")


# =============================================================================
# SPECIALIZED AI AGENTS
# =============================================================================

# -----------------------------------------------------------------------------
# AGENT 1: Mental Health Awareness Agent
# -----------------------------------------------------------------------------
def awareness_agent(query: str) -> dict:
    """
    Agent 1: Mental Health Awareness Agent
    Educates users about mental health topics using IBM Granite + RAG context.

    Topics: Anxiety, Depression, Stress, Burnout, Mindfulness, Self-Care

    Args:
        query (str): User's question about mental health

    Returns:
        dict: Agent name, response, and sources used
    """
    logger.info("[Agent 1] Mental Health Awareness Agent activated.")

    # Retrieve relevant knowledge from RAG system
    rag_context = retrieve_context(query)
    rag_section = f"\n\nRelevant Knowledge Base Context:\n{rag_context}" if rag_context else ""

    # Construct IBM Granite prompt for awareness education
    prompt = f"""You are MindGuard AI's Mental Health Awareness Agent, powered by IBM Granite.
Your role is to provide accurate, compassionate, and evidence-based mental health education.

{rag_section}

User Question: {query}

Provide a clear, educational, and empathetic response covering:
1. A clear explanation of the topic
2. Common symptoms or indicators
3. Evidence-based insights
4. Practical tips or takeaways

Response:"""

    # --- IBM watsonx.ai Granite Model Call ---
    response = generate_response(prompt, max_tokens=400)

    return {
        "agent": "Mental Health Awareness Agent",
        "agent_id": 1,
        "icon": "🧠",
        "reason": "Query relates to mental health education and awareness topics.",
        "response": response,
        "rag_used": bool(rag_context),
        "sources": ["WHO Mental Health Guidelines", "Built-in Knowledge Base"] if rag_context else []
    }


# -----------------------------------------------------------------------------
# AGENT 2: Emotional Support Agent
# -----------------------------------------------------------------------------
def emotional_support_agent(query: str) -> dict:
    """
    Agent 2: Emotional Support Agent
    Provides empathetic, non-judgmental conversational support.
    Active listening, emotion acknowledgment, coping encouragement.

    Args:
        query (str): User's emotional expression or concern

    Returns:
        dict: Agent name, empathetic response, and disclaimer
    """
    logger.info("[Agent 2] Emotional Support Agent activated.")

    rag_context = retrieve_context(query)
    rag_section = f"\n\nSupportive Context:\n{rag_context}" if rag_context else ""

    # IBM Granite prompt for empathetic emotional support
    prompt = f"""You are MindGuard AI's Emotional Support Agent, powered by IBM Granite.
Your role is to provide warm, empathetic, non-judgmental emotional support to individuals
who are experiencing emotional distress, loneliness, stress, or difficult feelings.

{rag_section}

The user shares: {query}

Respond with:
1. Genuine acknowledgment of their feelings
2. Validation that their emotions are understandable
3. Gentle encouragement
4. One practical coping suggestion
5. A reminder that professional help is available

Keep the tone warm, human, and supportive. Avoid clinical language.

Response:"""

    # --- IBM watsonx.ai Granite Model Call ---
    response = generate_response(prompt, max_tokens=400)

    disclaimer = ("\n\n⚕️ MindGuard AI provides educational and emotional support. "
                  "It is not a substitute for professional medical or psychological care. "
                  "If you are in crisis, please contact a mental health professional immediately.")

    return {
        "agent": "Emotional Support Agent",
        "agent_id": 2,
        "icon": "💚",
        "reason": "User expressed emotional distress or is seeking empathetic support.",
        "response": response + disclaimer,
        "rag_used": bool(rag_context),
        "sources": []
    }


# -----------------------------------------------------------------------------
# AGENT 3: Distress Detection Agent
# -----------------------------------------------------------------------------
DISTRESS_KEYWORDS = {
    "high": ["suicide", "kill myself", "end my life", "don't want to live", "no reason to live",
             "better off dead", "hurt myself", "self-harm", "hopeless", "worthless", "can't go on",
             "ending it all", "nothing matters anymore", "disappear forever"],
    "moderate": ["depressed", "really sad", "crying all the time", "can't sleep", "panic attack",
                 "extreme anxiety", "overwhelmed", "breaking down", "losing control", "terrified",
                 "isolated", "nobody cares", "exhausted", "empty inside", "falling apart"],
    "low": ["stressed", "worried", "anxious", "sad", "lonely", "tired", "frustrated",
            "upset", "nervous", "down", "not okay", "struggling", "hard time"]
}

def detect_risk(text: str) -> dict:
    """
    Analyze text for mental health distress signals.
    Uses keyword analysis + IBM Granite reasoning for nuanced risk assessment.

    Args:
        text (str): User's message or journal entry

    Returns:
        dict: Risk level, score (0-100), indicators found, explanation
    """
    text_lower = text.lower()
    high_found     = [kw for kw in DISTRESS_KEYWORDS["high"]     if kw in text_lower]
    moderate_found = [kw for kw in DISTRESS_KEYWORDS["moderate"] if kw in text_lower]
    low_found      = [kw for kw in DISTRESS_KEYWORDS["low"]      if kw in text_lower]

    # Calculate preliminary risk score
    score = min(100, len(high_found) * 35 + len(moderate_found) * 15 + len(low_found) * 5)

    if high_found or score >= 60:
        risk_level = "High Risk"
        base_score = max(70, score)
    elif moderate_found or score >= 30:
        risk_level = "Moderate Risk"
        base_score = max(40, score)
    elif low_found or score >= 10:
        risk_level = "Low Risk"
        base_score = max(15, score)
    else:
        risk_level = "Minimal Risk"
        base_score = score

    all_indicators = high_found + moderate_found + low_found
    return {
        "risk_level": risk_level,
        "score": min(base_score, 100),
        "indicators": all_indicators[:8],
        "high_signals": high_found,
        "moderate_signals": moderate_found,
        "low_signals": low_found
    }


def distress_detection_agent(query: str) -> dict:
    """
    Agent 3: Distress Detection Agent
    Analyzes user input for mental health risk indicators.
    Classifies risk level and generates IBM Granite reasoning.

    Args:
        query (str): User message or journal entry

    Returns:
        dict: Risk assessment, score, explanation, next steps
    """
    logger.info("[Agent 3] Distress Detection Agent activated.")

    risk_data = detect_risk(query)
    risk_level = risk_data["risk_level"]
    risk_score = risk_data["score"]
    indicators = risk_data["indicators"]

    indicator_str = ", ".join(indicators) if indicators else "subtle emotional cues"

    # IBM Granite prompt for detailed distress reasoning
    prompt = f"""You are MindGuard AI's Distress Detection Agent, powered by IBM Granite.
Analyze the following user message for signs of mental health distress.

User Message: {query}

Preliminary Risk Assessment: {risk_level} (Score: {risk_score}/100)
Detected Indicators: {indicator_str}

Provide:
1. A compassionate explanation of the detected distress signals
2. What the risk classification means for this person
3. Three specific, actionable next steps they should take
4. A gentle, caring closing message

Be sensitive, non-alarmist, and supportive in tone.

Analysis:"""

    # --- IBM watsonx.ai Granite Model Call ---
    explanation = generate_response(prompt, max_tokens=400)

    return {
        "agent": "Distress Detection Agent",
        "agent_id": 3,
        "icon": "🔍",
        "reason": "Message analyzed for emotional distress signals and risk indicators.",
        "risk_level": risk_level,
        "risk_score": risk_score,
        "indicators": indicators,
        "response": explanation,
        "rag_used": False,
        "sources": []
    }


# -----------------------------------------------------------------------------
# AGENT 4: Prevention & Wellness Agent
# -----------------------------------------------------------------------------
def generate_wellness_plan(mood: str, stress_level: str, emotional_state: str) -> str:
    """
    Generate a personalized wellness plan using IBM Granite.

    Args:
        mood (str): Current mood description
        stress_level (str): Self-reported stress level
        emotional_state (str): Broader emotional context

    Returns:
        str: Personalized wellness plan from Granite
    """
    prompt = f"""You are MindGuard AI's Prevention & Wellness Agent, powered by IBM Granite.
Create a personalized daily wellness plan based on the user's current state.

Current Mood: {mood}
Stress Level: {stress_level}
Emotional State: {emotional_state}

Generate a warm, practical wellness plan that includes:
1. 🌬️ Morning Breathing Exercise (2-3 minutes)
2. 🧘 Mindfulness or Meditation Activity (5-10 minutes)
3. ✍️ Journaling Prompt tailored to their mood
4. 🚶 Physical Activity Recommendation
5. 😴 Sleep Hygiene Tip
6. 🤝 Social Connection Suggestion
7. 🌟 One Positive Affirmation

Make it personal, actionable, and encouraging.

Wellness Plan:"""

    # --- IBM watsonx.ai Granite Model Call ---
    return generate_response(prompt, max_tokens=500)


def wellness_agent(query: str) -> dict:
    """
    Agent 4: Prevention & Wellness Agent
    Generates personalized wellness plans and coping strategies.

    Args:
        query (str): User's mood, stress level, or wellness request

    Returns:
        dict: Personalized wellness plan and recommendations
    """
    logger.info("[Agent 4] Prevention & Wellness Agent activated.")

    # Extract mood signals from query
    q_lower = query.lower()
    if any(w in q_lower for w in ["stressed", "stress", "tense", "pressure"]):
        mood, stress, state = "stressed", "high", "under pressure"
    elif any(w in q_lower for w in ["anxious", "anxiety", "worried", "nervous"]):
        mood, stress, state = "anxious", "moderate to high", "worried about the future"
    elif any(w in q_lower for w in ["sad", "depressed", "down", "blue", "unhappy"]):
        mood, stress, state = "sad", "moderate", "emotionally low"
    elif any(w in q_lower for w in ["tired", "exhausted", "burnout", "drained"]):
        mood, stress, state = "exhausted", "high", "burned out"
    elif any(w in q_lower for w in ["okay", "fine", "alright", "good", "well"]):
        mood, stress, state = "neutral", "low", "generally okay"
    else:
        mood, stress, state = "uncertain", "moderate", "looking for guidance"

    wellness_plan = generate_wellness_plan(mood, stress, state)

    return {
        "agent": "Prevention & Wellness Agent",
        "agent_id": 4,
        "icon": "🌱",
        "reason": "User needs personalized wellness recommendations and coping strategies.",
        "mood_detected": mood,
        "stress_level": stress,
        "response": wellness_plan,
        "rag_used": False,
        "sources": []
    }


# -----------------------------------------------------------------------------
# AGENT 5: Human Support Connector Agent
# -----------------------------------------------------------------------------
SUPPORT_RESOURCES = {
    "crisis_helplines": [
        {"name": "988 Suicide & Crisis Lifeline (US)", "contact": "Call or text 988", "availability": "24/7"},
        {"name": "Crisis Text Line", "contact": "Text HOME to 741741", "availability": "24/7"},
        {"name": "International Association for Suicide Prevention", "contact": "https://www.iasp.info/resources/Crisis_Centres/", "availability": "Online"},
        {"name": "Befrienders Worldwide", "contact": "https://www.befrienders.org/", "availability": "Online"},
        {"name": "SAMHSA National Helpline (US)", "contact": "1-800-662-4357", "availability": "24/7, Free, Confidential"},
    ],
    "professional_resources": [
        {"name": "Psychology Today Therapist Finder", "contact": "https://www.psychologytoday.com/us/therapists", "type": "Therapist Directory"},
        {"name": "BetterHelp Online Therapy", "contact": "https://www.betterhelp.com/", "type": "Online Therapy"},
        {"name": "Open Path Collective (Affordable)", "contact": "https://openpathcollective.org/", "type": "Affordable Therapy"},
        {"name": "NAMI Helpline", "contact": "1-800-950-6264", "type": "Mental Health Alliance"},
    ],
    "support_groups": [
        {"name": "NAMI Support Groups", "contact": "https://www.nami.org/Support-Education/Support-Groups", "type": "Peer Support"},
        {"name": "Mental Health America", "contact": "https://www.mhanational.org/", "type": "Advocacy & Support"},
        {"name": "7 Cups (Free Online Support)", "contact": "https://www.7cups.com/", "type": "Peer Listening"},
    ]
}

def support_connector_agent(query: str, risk_level: str = "Low Risk") -> dict:
    """
    Agent 5: Human Support Connector Agent
    Recommends mental health resources, helplines, and professional support.
    Escalates recommendations based on detected risk level.

    Args:
        query (str): User's request or concern
        risk_level (str): Risk level from distress detection agent

    Returns:
        dict: Curated support resources, helplines, professional suggestions
    """
    logger.info(f"[Agent 5] Human Support Connector Agent activated. Risk: {risk_level}")

    is_high_risk = "high" in risk_level.lower()

    # IBM Granite prompt for personalized resource recommendation
    prompt = f"""You are MindGuard AI's Human Support Connector Agent, powered by IBM Granite.
Your role is to compassionately guide someone to appropriate professional mental health support.

User Situation: {query}
Detected Risk Level: {risk_level}

{'⚠️ HIGH RISK DETECTED: Prioritize immediate crisis resources.' if is_high_risk else ''}

Provide:
1. A warm, encouraging message about seeking professional help
2. Why professional support is valuable for their situation
3. How to take the first step in reaching out for help
4. Reassurance that seeking help is a sign of strength

Remind them: Professional help is always the right choice.

Response:"""

    # --- IBM watsonx.ai Granite Model Call ---
    connector_response = generate_response(prompt, max_tokens=350)

    # Select appropriate resources based on risk
    if is_high_risk:
        primary_resources = SUPPORT_RESOURCES["crisis_helplines"]
        resource_type = "🚨 Immediate Crisis Support Resources"
    else:
        primary_resources = SUPPORT_RESOURCES["professional_resources"]
        resource_type = "🤝 Professional Support Resources"

    return {
        "agent": "Human Support Connector Agent",
        "agent_id": 5,
        "icon": "🤝",
        "reason": "Connecting user with appropriate professional and community support resources.",
        "response": connector_response,
        "resource_type": resource_type,
        "primary_resources": primary_resources,
        "crisis_helplines": SUPPORT_RESOURCES["crisis_helplines"],
        "support_groups": SUPPORT_RESOURCES["support_groups"],
        "is_high_risk": is_high_risk,
        "rag_used": False,
        "sources": []
    }


# =============================================================================
# MASTER ORCHESTRATOR AGENT
# =============================================================================
def orchestrate_agents(query: str) -> dict:
    """
    Master Orchestrator Agent — The Brain of MindGuard AI
    Analyzes user input and intelligently routes to specialized agents.
    Can combine outputs from multiple agents for comprehensive responses.

    Decision Logic:
    - Awareness keywords   → Awareness Agent
    - Emotional distress   → Emotional Support + Distress Detection + Support Connector
    - Crisis/High risk     → All agents prioritizing Distress + Support Connector
    - Wellness request     → Wellness Agent + Support Connector
    - Support request      → Support Connector Agent

    Args:
        query (str): Raw user input

    Returns:
        dict: Orchestrated multi-agent response with all relevant outputs
    """
    logger.info(f"[Orchestrator] Processing query: {query[:80]}...")
    q_lower = query.lower()
    timestamp = datetime.now().strftime("%H:%M:%S")

    # ── Step 1: Always run Distress Detection (safety-first architecture) ──
    distress_result = distress_detection_agent(query)
    risk_level = distress_result["risk_level"]
    risk_score = distress_result["risk_score"]

    results = {
        "timestamp": timestamp,
        "query": query,
        "orchestrator_decision": "",
        "agents_activated": [],
        "distress": distress_result,
        "awareness": None,
        "emotional_support": None,
        "wellness": None,
        "support_connector": None,
        "primary_agent": None,
        "combined_response": ""
    }

    # ── Step 2: Route based on risk and intent ──

    # HIGH RISK: Activate all agents with crisis priority
    if "High Risk" in risk_level or risk_score >= 65:
        results["orchestrator_decision"] = (
            "⚠️ High distress signals detected. Activating full multi-agent support protocol: "
            "Distress Detection + Emotional Support + Human Support Connector."
        )
        emotional = emotional_support_agent(query)
        support   = support_connector_agent(query, risk_level)
        results["emotional_support"] = emotional
        results["support_connector"] = support
        results["agents_activated"]  = ["Distress Detection Agent", "Emotional Support Agent", "Human Support Connector Agent"]
        results["primary_agent"]     = emotional
        results["combined_response"] = emotional["response"]

    # AWARENESS INTENT: Educational queries
    elif any(kw in q_lower for kw in [
        "what is", "explain", "tell me about", "how does", "anxiety", "depression",
        "burnout", "mindfulness", "ptsd", "bipolar", "schizophrenia", "ocd",
        "mental health", "therapy", "cbt", "self-care", "stress management"
    ]):
        results["orchestrator_decision"] = (
            "Query identified as mental health education request. "
            "Activating Mental Health Awareness Agent with RAG knowledge retrieval."
        )
        awareness = awareness_agent(query)
        results["awareness"] = awareness
        results["agents_activated"] = ["Mental Health Awareness Agent"]
        results["primary_agent"] = awareness
        results["combined_response"] = awareness["response"]

    # WELLNESS INTENT: Coping and wellness queries
    elif any(kw in q_lower for kw in [
        "coping", "cope", "wellness", "exercise", "meditation", "breathe", "breathing",
        "sleep", "relax", "calm", "plan", "routine", "self-care", "feel better",
        "tips", "advice", "help me", "what can i do", "how to deal"
    ]):
        results["orchestrator_decision"] = (
            "Wellness and coping strategy request identified. "
            "Activating Prevention & Wellness Agent for personalized recommendations."
        )
        wellness = wellness_agent(query)
        results["wellness"] = wellness
        results["agents_activated"] = ["Prevention & Wellness Agent"]
        results["primary_agent"] = wellness
        results["combined_response"] = wellness["response"]

    # SUPPORT INTENT: Resource and helpline queries
    elif any(kw in q_lower for kw in [
        "help", "support", "therapist", "counseling", "psychiatrist", "hotline",
        "helpline", "crisis", "resource", "where can i", "who can i call"
    ]):
        results["orchestrator_decision"] = (
            "Support resource request detected. "
            "Activating Human Support Connector Agent."
        )
        support = support_connector_agent(query, risk_level)
        results["support_connector"] = support
        results["agents_activated"] = ["Human Support Connector Agent"]
        results["primary_agent"] = support
        results["combined_response"] = support["response"]

    # EMOTIONAL DISTRESS: Moderate risk or emotional expressions
    elif any(kw in q_lower for kw in [
        "feel", "feeling", "sad", "lonely", "anxious", "overwhelmed", "scared",
        "worried", "upset", "angry", "stressed", "tired", "exhausted", "hopeless",
        "frustrated", "lost", "empty", "numb", "broken", "struggling"
    ]) or "Moderate Risk" in risk_level:
        results["orchestrator_decision"] = (
            "Emotional distress expressed. Activating Emotional Support Agent + "
            "Wellness Agent for comprehensive support."
        )
        emotional = emotional_support_agent(query)
        wellness  = wellness_agent(query)
        results["emotional_support"] = emotional
        results["wellness"] = wellness
        results["agents_activated"] = ["Emotional Support Agent", "Prevention & Wellness Agent"]
        results["primary_agent"] = emotional
        results["combined_response"] = emotional["response"]

    # DEFAULT: General query — Awareness + Emotional Support
    else:
        results["orchestrator_decision"] = (
            "General mental health query. Activating Mental Health Awareness Agent "
            "and Emotional Support Agent for a well-rounded response."
        )
        awareness  = awareness_agent(query)
        emotional  = emotional_support_agent(query)
        results["awareness"]        = awareness
        results["emotional_support"] = emotional
        results["agents_activated"] = ["Mental Health Awareness Agent", "Emotional Support Agent"]
        results["primary_agent"]    = awareness
        results["combined_response"] = awareness["response"]

    # Always include support connector for moderate or high risk
    if risk_score >= 40 and results["support_connector"] is None:
        results["support_connector"] = support_connector_agent(query, risk_level)
        if "Human Support Connector Agent" not in results["agents_activated"]:
            results["agents_activated"].append("Human Support Connector Agent")

    logger.info(f"[Orchestrator] Agents activated: {results['agents_activated']}")
    return results


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serve the MindGuard AI single-page application."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint — orchestrates all agents for a given user query.
    POST body: { "message": "user input text" }
    Returns: Full orchestrated multi-agent response JSON
    """
    data  = request.get_json(silent=True) or {}
    query = data.get("message", "").strip()
    if not query:
        return jsonify({"error": "Message cannot be empty."}), 400
    try:
        result = orchestrate_agents(query)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    RAG document upload endpoint.
    Accepts PDF or TXT files and ingests them into the knowledge base.
    POST: multipart/form-data with 'file' field
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    file = request.files["file"]
    filename = file.filename or "uploaded_file"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        if ext == "txt":
            text = file.read().decode("utf-8", errors="ignore")
        elif ext == "pdf" and PDF_SUPPORT:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif ext == "pdf" and not PDF_SUPPORT:
            return jsonify({"error": "PyPDF2 not installed. Only TXT files supported."}), 400
        else:
            return jsonify({"error": "Only PDF and TXT files are supported."}), 400

        if len(text.strip()) < 50:
            return jsonify({"error": "Document appears to be empty or unreadable."}), 400

        ingest_document(text, source=filename)
        return jsonify({
            "success": True,
            "message": f"Document '{filename}' successfully ingested into RAG knowledge base.",
            "chunks": len(rag_documents),
            "filename": filename
        })
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """System status endpoint — returns watsonx.ai connection and RAG stats."""
    return jsonify({
        "status": "online",
        "watsonx_active": WATSONX_ACTIVE,
        "granite_model": GRANITE_MODEL_ID,
        "rag_documents": len(rag_documents),
        "pdf_support": PDF_SUPPORT,
        "mode": "IBM watsonx.ai (Live)" if WATSONX_ACTIVE else "Demo Mode",
        "agents": [
            "Mental Health Awareness Agent",
            "Emotional Support Agent",
            "Distress Detection Agent",
            "Prevention & Wellness Agent",
            "Human Support Connector Agent",
            "Master Orchestrator Agent"
        ]
    })


# =============================================================================
# HTML TEMPLATE — Modern Bootstrap 5 Single-Page Application
# =============================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MindGuard AI – Mental Health Awareness Agent</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
  :root {
    --primary: #4f7ef8;
    --primary-dark: #3563e9;
    --secondary: #7c5cd8;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg: #f0f4ff;
    --surface: #ffffff;
    --surface2: #f8f9ff;
    --border: #e2e8f0;
    --text: #1e293b;
    --muted: #64748b;
    --radius: 16px;
    --shadow: 0 4px 24px rgba(79,126,248,0.10);
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.08);
  }
  * { box-sizing: border-box; }
  body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .navbar-brand { font-weight: 700; font-size: 1.3rem; color: #fff !important; }
  .navbar { background: linear-gradient(135deg, #3563e9 0%, #7c5cd8 100%); padding: 12px 0; }
  .badge-watsonx { background: rgba(255,255,255,0.2); color: #fff; border-radius: 20px; font-size: 0.72rem; padding: 3px 10px; font-weight: 500; }
  .hero { background: linear-gradient(135deg, #3563e9 0%, #7c5cd8 100%); color: #fff; padding: 36px 0 28px; margin-bottom: 28px; }
  .hero h1 { font-size: 2rem; font-weight: 700; margin-bottom: 8px; }
  .hero p { font-size: 1.05rem; opacity: 0.88; max-width: 600px; }
  .card { border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); background: var(--surface); }
  .card-header { border-radius: calc(var(--radius) - 1px) calc(var(--radius) - 1px) 0 0 !important; font-weight: 600; font-size: 0.95rem; border-bottom: 1px solid var(--border); padding: 14px 18px; }
  .chat-header { background: linear-gradient(90deg, #3563e9, #7c5cd8); color: #fff; }
  .agent-header { background: linear-gradient(90deg, #0ea5e9, #3563e9); color: #fff; }
  .risk-header { background: linear-gradient(90deg, #ef4444, #f59e0b); color: #fff; }
  .wellness-header { background: linear-gradient(90deg, #22c55e, #0ea5e9); color: #fff; }
  .support-header { background: linear-gradient(90deg, #7c5cd8, #3563e9); color: #fff; }
  .chat-messages { height: 420px; overflow-y: auto; padding: 18px; background: var(--surface2); border-radius: 0; }
  .chat-messages::-webkit-scrollbar { width: 5px; }
  .chat-messages::-webkit-scrollbar-track { background: #f1f5f9; }
  .chat-messages::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 10px; }
  .msg { display: flex; margin-bottom: 14px; align-items: flex-end; gap: 10px; }
  .msg.user { flex-direction: row-reverse; }
  .msg-bubble { max-width: 78%; padding: 12px 16px; border-radius: 18px; font-size: 0.93rem; line-height: 1.6; }
  .msg.user .msg-bubble { background: linear-gradient(135deg, #3563e9, #7c5cd8); color: #fff; border-bottom-right-radius: 6px; }
  .msg.ai .msg-bubble { background: var(--surface); border: 1px solid var(--border); color: var(--text); border-bottom-left-radius: 6px; box-shadow: var(--shadow-sm); }
  .msg-avatar { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0; }
  .msg.user .msg-avatar { background: #7c5cd8; color: #fff; }
  .msg.ai .msg-avatar { background: #e0e7ff; color: #3563e9; }
  .chat-input-area { padding: 14px; background: #fff; border-top: 1px solid var(--border); border-radius: 0 0 var(--radius) var(--radius); }
  .chat-input { border-radius: 30px; border: 1.5px solid var(--border); padding: 10px 20px; font-size: 0.94rem; transition: border 0.2s; }
  .chat-input:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(79,126,248,0.12); outline: none; }
  .btn-send { border-radius: 30px; background: linear-gradient(135deg, #3563e9, #7c5cd8); border: none; color: #fff; padding: 10px 24px; font-weight: 600; font-size: 0.94rem; transition: opacity 0.2s; }
  .btn-send:hover { opacity: 0.88; }
  .agent-pill { display: inline-flex; align-items: center; gap: 6px; background: #e0e7ff; color: #3563e9; border-radius: 20px; padding: 4px 12px; font-size: 0.8rem; font-weight: 600; margin: 3px 2px; }
  .agent-pill.active { background: linear-gradient(135deg, #3563e9, #7c5cd8); color: #fff; }
  .agent-card { border: 1.5px solid var(--border); border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; transition: all 0.25s; background: var(--surface2); }
  .agent-card.active { border-color: var(--primary); background: #f0f5ff; box-shadow: 0 0 0 3px rgba(79,126,248,0.10); }
  .agent-card .agent-name { font-weight: 600; font-size: 0.9rem; color: var(--text); }
  .agent-card .agent-desc { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }
  .risk-badge { display: inline-block; padding: 5px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
  .risk-minimal { background: #dcfce7; color: #166534; }
  .risk-low { background: #dbeafe; color: #1e40af; }
  .risk-moderate { background: #fef3c7; color: #92400e; }
  .risk-high { background: #fee2e2; color: #991b1b; }
  .risk-meter { height: 10px; border-radius: 10px; background: #e2e8f0; margin: 10px 0; overflow: hidden; }
  .risk-fill { height: 100%; border-radius: 10px; transition: width 0.8s ease; }
  .resource-item { background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }
  .resource-item .r-name { font-weight: 600; font-size: 0.88rem; }
  .resource-item .r-contact { font-size: 0.8rem; color: var(--primary); }
  .resource-item .r-avail { font-size: 0.75rem; color: var(--muted); }
  .crisis-banner { background: linear-gradient(135deg, #fee2e2, #fef3c7); border: 2px solid #ef4444; border-radius: 12px; padding: 14px 16px; margin-bottom: 14px; }
  .wellness-item { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.9rem; }
  .workflow-step { display: flex; align-items: center; gap: 12px; padding: 10px 14px; background: var(--surface2); border-radius: 10px; margin-bottom: 8px; border: 1px solid var(--border); }
  .step-num { width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg, #3563e9, #7c5cd8); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; flex-shrink: 0; }
  .step-text { font-size: 0.86rem; color: var(--text); }
  .step-text strong { display: block; font-size: 0.82rem; color: var(--muted); font-weight: 400; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }
  .status-live { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
  .status-demo { background: #f59e0b; }
  .upload-zone { border: 2px dashed var(--border); border-radius: 12px; padding: 24px; text-align: center; cursor: pointer; transition: all 0.2s; background: var(--surface2); }
  .upload-zone:hover { border-color: var(--primary); background: #f0f5ff; }
  .upload-zone .upload-icon { font-size: 2.2rem; margin-bottom: 8px; }
  .suggestion-btn { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 6px 14px; font-size: 0.82rem; color: var(--primary); cursor: pointer; transition: all 0.2s; margin: 3px; display: inline-block; }
  .suggestion-btn:hover { background: #e0e7ff; border-color: var(--primary); }
  .typing-indicator { display: none; }
  .typing-indicator span { display: inline-block; width: 7px; height: 7px; background: var(--primary); border-radius: 50%; margin: 0 2px; animation: bounce 1.2s infinite; }
  .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
  .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-8px)} }
  .panel-empty { text-align: center; padding: 28px 16px; color: var(--muted); font-size: 0.9rem; }
  .panel-empty .empty-icon { font-size: 2.5rem; margin-bottom: 10px; opacity: 0.5; }
  .section-title { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }
  footer { background: #1e293b; color: #94a3b8; text-align: center; padding: 20px 0; font-size: 0.82rem; margin-top: 48px; }
  footer a { color: #7c5cd8; text-decoration: none; }
  @media (max-width: 768px) { .hero h1 { font-size: 1.4rem; } }
</style>
</head>
<body>

<!-- ═══════════════════════════════════ NAVBAR ═══════════════════════════════════ -->
<nav class="navbar">
  <div class="container-fluid px-4">
    <span class="navbar-brand">🧠 MindGuard AI</span>
    <div class="d-flex align-items-center gap-2 ms-auto">
      <span class="badge-watsonx">⚡ IBM watsonx.ai</span>
      <span class="badge-watsonx">🪨 Granite Models</span>
      <span id="statusBadge" class="badge-watsonx">⏳ Connecting...</span>
    </div>
  </div>
</nav>

<!-- ════════════════════════════════════ HERO ════════════════════════════════════ -->
<div class="hero">
  <div class="container">
    <h1>🛡️ MindGuard AI</h1>
    <p>An Agentic AI platform for Mental Health Awareness & Suicide Prevention — powered by IBM Granite Models and IBM watsonx.ai Studio.</p>
    <div class="d-flex flex-wrap gap-2 mt-3">
      <span class="badge-watsonx">🤖 Multi-Agent Orchestration</span>
      <span class="badge-watsonx">📚 RAG Knowledge Base</span>
      <span class="badge-watsonx">🔍 Distress Detection</span>
      <span class="badge-watsonx">💚 Emotional Support</span>
      <span class="badge-watsonx">🌱 Wellness Planning</span>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════ MAIN LAYOUT ═══════════════════════════════ -->
<div class="container mb-5">
  <div class="row g-4">

    <!-- ╔══════════════════════════ LEFT COLUMN ══════════════════════════╗ -->
    <div class="col-lg-5">

      <!-- CHAT INTERFACE -->
      <div class="card mb-4">
        <div class="card-header chat-header d-flex align-items-center gap-2">
          💬 Mental Health Chat
          <span class="ms-auto badge bg-white text-primary" style="font-size:0.73rem;">MindGuard AI</span>
        </div>
        <div class="chat-messages" id="chatMessages">
          <div class="msg ai">
            <div class="msg-avatar">🧠</div>
            <div class="msg-bubble">
              Hello! I'm <strong>MindGuard AI</strong>, your compassionate mental health companion powered by <strong>IBM Granite</strong> on <strong>IBM watsonx.ai</strong>.<br><br>
              I'm here to support your mental wellness journey. You can ask me about mental health topics, share how you're feeling, or request coping strategies.<br><br>
              <em>How are you feeling today?</em>
            </div>
          </div>
        </div>
        <!-- Typing Indicator -->
        <div class="px-3 pb-1" style="background: var(--surface2);">
          <div class="typing-indicator" id="typingIndicator">
            <div class="msg-bubble" style="display:inline-block;padding:8px 16px;">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
        <!-- Input Area -->
        <div class="chat-input-area">
          <div class="d-flex gap-2">
            <input type="text" id="chatInput" class="form-control chat-input" placeholder="Share how you're feeling..." />
            <button class="btn btn-send" onclick="sendMessage()">Send</button>
          </div>
          <!-- Quick Suggestions -->
          <div class="mt-2">
            <span class="suggestion-btn" onclick="useSuggestion('What is anxiety?')">What is anxiety?</span>
            <span class="suggestion-btn" onclick="useSuggestion('I feel overwhelmed and stressed')">I feel overwhelmed</span>
            <span class="suggestion-btn" onclick="useSuggestion('How can mindfulness help me?')">Mindfulness tips</span>
            <span class="suggestion-btn" onclick="useSuggestion('I need coping strategies for depression')">Coping strategies</span>
          </div>
        </div>
      </div>

      <!-- DOCUMENT UPLOAD (RAG) -->
      <div class="card">
        <div class="card-header" style="background:#f1f5f9;color:var(--text);">
          📚 RAG Knowledge Base
        </div>
        <div class="card-body p-3">
          <p class="text-muted" style="font-size:0.83rem;">Upload mental health documents (PDF/TXT) to enhance AI responses.</p>
          <label class="upload-zone w-100 d-block" for="docUpload">
            <div class="upload-icon">📄</div>
            <div style="font-size:0.88rem;font-weight:600;">Drop PDF or TXT here</div>
            <div style="font-size:0.78rem;color:var(--muted);">WHO Guidelines, Coping Resources, Mental Health Documents</div>
          </label>
          <input type="file" id="docUpload" accept=".pdf,.txt" style="display:none;" onchange="uploadDocument()"/>
          <div id="uploadStatus" class="mt-2" style="font-size:0.83rem;"></div>
          <div class="mt-2 d-flex align-items-center gap-2">
            <span style="font-size:0.8rem;color:var(--muted);">Knowledge chunks loaded:</span>
            <span id="ragCount" class="badge bg-primary">—</span>
          </div>
        </div>
      </div>
    </div>

    <!-- ╔══════════════════════════ RIGHT COLUMN ══════════════════════════╗ -->
    <div class="col-lg-7">

      <!-- AGENT ORCHESTRATION PANEL -->
      <div class="card mb-4">
        <div class="card-header agent-header d-flex align-items-center gap-2">
          🤖 Agent Orchestration Panel
          <span class="ms-auto" style="font-size:0.78rem;opacity:0.85;">Agentic AI Architecture</span>
        </div>
        <div class="card-body p-3" id="agentPanel">

          <!-- Agent Cards (static display) -->
          <div class="section-title">Specialized Agents</div>
          <div class="row g-2 mb-3" id="agentCards">
            <div class="col-6"><div class="agent-card" id="card-1"><div class="d-flex gap-2 align-items-start"><span style="font-size:1.3rem;">🧠</span><div><div class="agent-name">Awareness Agent</div><div class="agent-desc">Mental health education & topics</div></div></div></div></div>
            <div class="col-6"><div class="agent-card" id="card-2"><div class="d-flex gap-2 align-items-start"><span style="font-size:1.3rem;">💚</span><div><div class="agent-name">Emotional Support</div><div class="agent-desc">Empathetic listening & comfort</div></div></div></div></div>
            <div class="col-6"><div class="agent-card" id="card-3"><div class="d-flex gap-2 align-items-start"><span style="font-size:1.3rem;">🔍</span><div><div class="agent-name">Distress Detection</div><div class="agent-desc">Risk analysis & assessment</div></div></div></div></div>
            <div class="col-6"><div class="agent-card" id="card-4"><div class="d-flex gap-2 align-items-start"><span style="font-size:1.3rem;">🌱</span><div><div class="agent-name">Wellness Agent</div><div class="agent-desc">Personalized wellness plans</div></div></div></div></div>
            <div class="col-12"><div class="agent-card" id="card-5"><div class="d-flex gap-2 align-items-start"><span style="font-size:1.3rem;">🤝</span><div><div class="agent-name">Human Support Connector</div><div class="agent-desc">Professional resources & helplines</div></div></div></div></div>
          </div>

          <!-- Orchestrator Decision -->
          <div class="section-title">Orchestrator Decision</div>
          <div id="orchestratorDecision" class="panel-empty">
            <div class="empty-icon">🎯</div>
            <div>Send a message to see the orchestrator in action</div>
          </div>

          <!-- Workflow Steps -->
          <div id="workflowSteps" style="display:none;">
            <div class="section-title mt-3">Workflow Execution</div>
            <div id="workflowList"></div>
          </div>
        </div>
      </div>

      <!-- RISK DETECTION PANEL -->
      <div class="card mb-4">
        <div class="card-header risk-header d-flex align-items-center gap-2">
          🔍 Risk Detection Dashboard
          <span class="ms-auto" style="font-size:0.78rem;opacity:0.85;">Powered by IBM Granite</span>
        </div>
        <div class="card-body p-3" id="riskPanel">
          <div class="panel-empty">
            <div class="empty-icon">🛡️</div>
            <div>Risk assessment will appear here after analysis</div>
          </div>
        </div>
      </div>

      <!-- WELLNESS PANEL -->
      <div class="card mb-4">
        <div class="card-header wellness-header d-flex align-items-center gap-2">
          🌱 Wellness Recommendations
          <span class="ms-auto" style="font-size:0.78rem;opacity:0.85;">Personalized Plans</span>
        </div>
        <div class="card-body p-3" id="wellnessPanel">
          <div class="panel-empty">
            <div class="empty-icon">🌿</div>
            <div>Personalized wellness plan will appear here</div>
          </div>
        </div>
      </div>

      <!-- SUPPORT RESOURCES PANEL -->
      <div class="card">
        <div class="card-header support-header d-flex align-items-center gap-2">
          🤝 Support Resources
          <span class="ms-auto" style="font-size:0.78rem;opacity:0.85;">Professional Help</span>
        </div>
        <div class="card-body p-3" id="supportPanel">
          <div class="panel-empty">
            <div class="empty-icon">💼</div>
            <div>Professional support resources will appear here</div>
          </div>
        </div>
      </div>

    </div>
  </div>
</div>

<!-- FOOTER -->
<footer>
  <div class="container">
    <p>🧠 <strong>MindGuard AI</strong> — Mental Health Awareness & Suicide Prevention Agent</p>
    <p>Built with <a href="https://www.ibm.com/watsonx">IBM watsonx.ai Studio</a> · IBM Granite Models · Agentic AI Architecture</p>
    <p style="margin-top:6px;opacity:0.7;">⚕️ This application provides educational and emotional support only. It is NOT a substitute for professional medical or psychological care.</p>
    <p style="margin-top:10px;font-size:0.75rem;opacity:0.5;border-top:1px solid #334155;padding-top:10px;">Made with IBM Bob</p>
  </div>
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
// ═══════════════════════════════════════════════════════════════════
// MindGuard AI — Frontend JavaScript
// ═══════════════════════════════════════════════════════════════════

// Fetch system status on load
async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const badge = document.getElementById('statusBadge');
    if (d.watsonx_active) {
      badge.innerHTML = '<span class="status-dot status-live"></span>IBM watsonx.ai Live';
    } else {
      badge.innerHTML = '<span class="status-dot status-demo"></span>Demo Mode';
    }
    document.getElementById('ragCount').textContent = d.rag_documents;
  } catch(e) {
    document.getElementById('statusBadge').textContent = '⚠️ Offline';
  }
}
fetchStatus();

// Send user message
async function sendMessage() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';

  appendMessage(msg, 'user');
  showTyping(true);

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg})
    });
    const data = await r.json();
    showTyping(false);
    if (data.error) {
      appendMessage('⚠️ ' + data.error, 'ai');
      return;
    }
    // Display primary response in chat
    const primaryResponse = data.primary_agent ? data.primary_agent.response : data.combined_response;
    appendMessage(primaryResponse, 'ai');

    // Update all panels
    updateAgentPanel(data);
    updateRiskPanel(data.distress);
    updateWellnessPanel(data.wellness);
    updateSupportPanel(data.support_connector);

  } catch(e) {
    showTyping(false);
    appendMessage('⚠️ Connection error. Please check your server.', 'ai');
  }
}

// Allow Enter key to send
document.getElementById('chatInput').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') sendMessage();
});

// Use suggestion chip
function useSuggestion(text) {
  document.getElementById('chatInput').value = text;
  sendMessage();
}

// Append message to chat
function appendMessage(text, role) {
  const container = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const avatar = role === 'user' ? '👤' : '🧠';
  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-bubble">${escapeHtml(text).replace(/\\n/g,'<br/>')}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function showTyping(show) {
  document.getElementById('typingIndicator').style.display = show ? 'block' : 'none';
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Update Agent Orchestration Panel ──
function updateAgentPanel(data) {
  // Reset all cards
  for(let i=1;i<=5;i++) {
    document.getElementById('card-'+i).classList.remove('active');
  }
  // Highlight active agents
  const agentMap = {
    'Mental Health Awareness Agent': 1,
    'Emotional Support Agent': 2,
    'Distress Detection Agent': 3,
    'Prevention & Wellness Agent': 4,
    'Human Support Connector Agent': 5
  };
  (data.agents_activated || []).forEach(name => {
    const id = agentMap[name];
    if (id) document.getElementById('card-'+id).classList.add('active');
  });

  // Orchestrator decision
  const decDiv = document.getElementById('orchestratorDecision');
  decDiv.className = '';
  decDiv.innerHTML = `
    <div class="d-flex gap-2 align-items-start p-2 rounded" style="background:#f0f5ff;border:1px solid #c7d2fe;">
      <span style="font-size:1.3rem;">🎯</span>
      <div>
        <div style="font-weight:600;font-size:0.88rem;color:#3563e9;margin-bottom:4px;">Orchestrator Decision</div>
        <div style="font-size:0.85rem;color:var(--text);">${escapeHtml(data.orchestrator_decision || '')}</div>
        <div class="mt-2">${(data.agents_activated||[]).map(a=>`<span class="agent-pill active">${a}</span>`).join('')}</div>
      </div>
    </div>`;

  // Workflow steps
  const steps = ['User Query Received', 'Orchestrator Analysis', 'Agent Routing', 'Granite AI Processing', 'Response Compiled'];
  const stepDetails = [
    data.query ? `"${data.query.substring(0,50)}${data.query.length>50?'...':''}"` : '',
    'Risk & intent classification',
    (data.agents_activated||[]).join(', '),
    'IBM Granite Model inference',
    `${(data.agents_activated||[]).length} agent(s) responded`
  ];
  const wfList = document.getElementById('workflowList');
  wfList.innerHTML = steps.map((s,i)=>`
    <div class="workflow-step">
      <div class="step-num">${i+1}</div>
      <div class="step-text">${s}<strong>${stepDetails[i]}</strong></div>
    </div>`).join('');
  document.getElementById('workflowSteps').style.display = 'block';
}

// ── Update Risk Detection Panel ──
function updateRiskPanel(distress) {
  if (!distress) return;
  const panel = document.getElementById('riskPanel');
  const level = distress.risk_level || 'Minimal Risk';
  const score = distress.risk_score || 0;
  const levelClass = {
    'Minimal Risk': 'risk-minimal',
    'Low Risk': 'risk-low',
    'Moderate Risk': 'risk-moderate',
    'High Risk': 'risk-high'
  }[level] || 'risk-low';
  const fillColor = {
    'Minimal Risk': '#22c55e',
    'Low Risk': '#3b82f6',
    'Moderate Risk': '#f59e0b',
    'High Risk': '#ef4444'
  }[level] || '#3b82f6';

  const indicators = (distress.indicators || []).map(i => `<span class="badge bg-secondary me-1 mb-1" style="font-size:0.73rem;">${escapeHtml(i)}</span>`).join('');

  panel.innerHTML = `
    <div class="d-flex align-items-center gap-3 mb-3">
      <div>
        <div class="section-title mb-1">Risk Level</div>
        <span class="risk-badge ${levelClass}">${level}</span>
      </div>
      <div class="flex-grow-1">
        <div class="section-title mb-1">Risk Score</div>
        <div class="d-flex align-items-center gap-2">
          <div class="risk-meter flex-grow-1"><div class="risk-fill" style="width:${score}%;background:${fillColor};"></div></div>
          <span style="font-weight:700;font-size:1.1rem;color:${fillColor};">${score}/100</span>
        </div>
      </div>
    </div>
    ${indicators ? `<div class="mb-3"><div class="section-title">Detected Indicators</div>${indicators}</div>` : ''}
    ${distress.response ? `<div class="section-title">AI Analysis (IBM Granite)</div><div style="font-size:0.86rem;line-height:1.7;color:var(--text);background:var(--surface2);padding:12px;border-radius:10px;border:1px solid var(--border);">${escapeHtml(distress.response).replace(/\\n/g,'<br/>')}</div>` : ''}
    ${level === 'High Risk' ? `<div class="crisis-banner mt-3"><strong>⚠️ Immediate Support Recommended</strong><p style="margin:4px 0 0;font-size:0.85rem;">Please contact a crisis helpline immediately. <strong>988</strong> (US) or text HOME to <strong>741741</strong>.</p></div>` : ''}`;
}

// ── Update Wellness Panel ──
function updateWellnessPanel(wellness) {
  const panel = document.getElementById('wellnessPanel');
  if (!wellness) {
    panel.innerHTML = '<div class="panel-empty"><div class="empty-icon">🌿</div><div>Wellness plan will appear when relevant</div></div>';
    return;
  }
  panel.innerHTML = `
    <div class="d-flex align-items-center gap-2 mb-3">
      <span style="font-size:1.4rem;">🌱</span>
      <div>
        <div style="font-weight:600;font-size:0.95rem;">Personalized Wellness Plan</div>
        <div style="font-size:0.78rem;color:var(--muted);">Mood: ${escapeHtml(wellness.mood_detected||'Unknown')} · Stress: ${escapeHtml(wellness.stress_level||'Unknown')}</div>
      </div>
    </div>
    <div style="font-size:0.88rem;line-height:1.75;color:var(--text);background:#f0fdf4;padding:14px;border-radius:10px;border:1px solid #bbf7d0;">${escapeHtml(wellness.response||'').replace(/\\n/g,'<br/>')}</div>`;
}

// ── Update Support Resources Panel ──
function updateSupportPanel(support) {
  const panel = document.getElementById('supportPanel');
  if (!support) {
    panel.innerHTML = '<div class="panel-empty"><div class="empty-icon">💼</div><div>Resources appear when support is recommended</div></div>';
    return;
  }
  const crisisHTML = (support.crisis_helplines || []).slice(0,3).map(r => `
    <div class="resource-item">
      <div class="r-name">📞 ${escapeHtml(r.name)}</div>
      <div class="r-contact">${escapeHtml(r.contact)}</div>
      <div class="r-avail">🕐 ${escapeHtml(r.availability||'')}</div>
    </div>`).join('');

  const primaryHTML = (support.primary_resources || []).slice(0,3).map(r => `
    <div class="resource-item">
      <div class="r-name">🏥 ${escapeHtml(r.name)}</div>
      <div class="r-contact">${escapeHtml(r.contact)}</div>
      <div class="r-avail">${escapeHtml(r.type||'')}</div>
    </div>`).join('');

  panel.innerHTML = `
    ${support.is_high_risk ? `<div class="crisis-banner"><strong>🚨 High Risk — Immediate Help Available</strong><p style="margin:4px 0 0;font-size:0.85rem;">Please reach out immediately. You are not alone.</p></div>` : ''}
    ${support.response ? `<div style="font-size:0.88rem;line-height:1.7;background:var(--surface2);padding:12px;border-radius:10px;border:1px solid var(--border);margin-bottom:12px;">${escapeHtml(support.response).replace(/\\n/g,'<br/>')}</div>` : ''}
    <div class="section-title">🚨 Crisis Helplines (24/7)</div>
    ${crisisHTML}
    <div class="section-title mt-3">${escapeHtml(support.resource_type||'Professional Resources')}</div>
    ${primaryHTML}`;
}

// ── Upload Document to RAG ──
async function uploadDocument() {
  const fileInput = document.getElementById('docUpload');
  const statusDiv = document.getElementById('uploadStatus');
  if (!fileInput.files.length) return;
  const file = fileInput.files[0];
  statusDiv.innerHTML = `<span style="color:var(--primary);">⏳ Uploading ${file.name}...</span>`;
  const formData = new FormData();
  formData.append('file', file);
  try {
    const r = await fetch('/api/upload', {method:'POST', body: formData});
    const d = await r.json();
    if (d.success) {
      statusDiv.innerHTML = `<span style="color:green;">✅ ${d.message}</span>`;
      document.getElementById('ragCount').textContent = d.chunks;
    } else {
      statusDiv.innerHTML = `<span style="color:red;">❌ ${d.error}</span>`;
    }
  } catch(e) {
    statusDiv.innerHTML = `<span style="color:red;">❌ Upload failed</span>`;
  }
  fileInput.value = '';
}
</script>
</body>
</html>"""


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  🧠 MindGuard AI – Mental Health Awareness & Prevention Agent")
    print("=" * 65)
    print(f"  IBM watsonx.ai Active : {WATSONX_ACTIVE}")
    print(f"  Granite Model         : {GRANITE_MODEL_ID}")
    print(f"  RAG Documents Loaded  : {len(rag_documents)} chunks")
    print(f"  PDF Support           : {PDF_SUPPORT}")
    print(f"  Mode                  : {'Live (IBM watsonx.ai)' if WATSONX_ACTIVE else 'Demo (No credentials)'}")
    print("=" * 65)
    print("  Agents Ready:")
    print("    1. Mental Health Awareness Agent   🧠")
    print("    2. Emotional Support Agent          💚")
    print("    3. Distress Detection Agent         🔍")
    print("    4. Prevention & Wellness Agent      🌱")
    print("    5. Human Support Connector Agent    🤝")
    print("    6. Master Orchestrator Agent        🎯")
    print("=" * 65)
    print("  To enable IBM watsonx.ai Granite:")
    print("    set WATSONX_API_KEY=your-api-key")
    print("    set WATSONX_PROJECT_ID=your-project-id")
    print("    set WATSONX_URL=https://us-south.ml.cloud.ibm.com")
    print("=" * 65)
    print("  Starting server at http://localhost:5000")
    print("=" * 65)
    app.run(debug=True, host="0.0.0.0", port=5000)
