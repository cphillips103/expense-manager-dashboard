import os
import logging
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from google.adk.sessions import VertexAiSessionService
import vertexai
from vertexai.preview.reasoning_engines import ReasoningEngine
from google.cloud.aiplatform_v1beta1 import types as aip_types
from vertexai.reasoning_engines import _utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manager_dashboard")

app = FastAPI(title="Manager Approval Dashboard")

# Read env variables
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
AGENT_RUNTIME_ID = os.environ.get("AGENT_RUNTIME_ID")

session_service = None
remote_agent = None

def get_clients():
    global session_service, remote_agent
    if session_service is None or remote_agent is None:
        proj = os.environ.get("GOOGLE_CLOUD_PROJECT") or PROJECT_ID
        runtime_id = os.environ.get("AGENT_RUNTIME_ID") or AGENT_RUNTIME_ID
        
        if not proj or not runtime_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT and AGENT_RUNTIME_ID environment variables must be set.")
        
        if "/" in runtime_id:
            short_id = runtime_id.split("/")[-1]
        else:
            short_id = runtime_id
            
        logger.info(f"Initializing clients with Project: {proj}, Location: us-east1, Engine: {short_id}")
        vertexai.init(project=proj, location="us-east1")
        
        session_service = VertexAiSessionService(
            project=proj,
            location="us-east1",
            agent_engine_id=short_id
        )
        
        full_resource_name = f"projects/{proj}/locations/us-east1/reasoningEngines/{short_id}"
        remote_agent = ReasoningEngine(full_resource_name)
        
    return session_service, remote_agent

@app.get("/api/pending")
async def get_pending_approvals():
    try:
        svc, _ = get_clients()
        logger.info("Fetching sessions list...")
        response = await svc.list_sessions(app_name="app")
        sessions = response.sessions
        logger.info(f"Found {len(sessions)} sessions. Checking event histories...")
        
        pending_list = []
        
        for s in sessions:
            try:
                # Fetch full session details with history/events
                full_session = await svc.get_session(
                    app_name="app",
                    user_id=s.user_id,
                    session_id=s.id
                )
                
                if not full_session or not full_session.events:
                    continue
                
                pending_calls = {}
                
                for event in full_session.events:
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            # Check for function call with name='adk_request_input'
                            fc = getattr(part, "function_call", None)
                            if fc and fc.name == "adk_request_input":
                                args = fc.args or {}
                                interrupt_id = args.get("interruptId") or fc.id or "manager_review"
                                pending_calls[interrupt_id] = {
                                    "interrupt_id": interrupt_id,
                                    "message": args.get("message", ""),
                                    "timestamp": event.timestamp
                                }
                            
                            # Check for function response resolving the call
                            fr = getattr(part, "function_response", None)
                            if fr:
                                if fr.id in pending_calls:
                                    pending_calls.pop(fr.id)
                                elif fr.name in pending_calls:
                                    pending_calls.pop(fr.name)
                
                if pending_calls:
                    expense_report = full_session.state.get("expense_report") or {}
                    amount = expense_report.get("amount", 0.0)
                    description = expense_report.get("description", "Unknown")
                    merchant = expense_report.get("merchant", "")
                    
                    for interrupt_id, call_info in pending_calls.items():
                        pending_list.append({
                            "session_id": s.id,
                            "user_id": s.user_id,
                            "interrupt_id": interrupt_id,
                            "message": call_info["message"],
                            "timestamp": call_info["timestamp"],
                            "expense": {
                                "amount": amount,
                                "description": description,
                                "merchant": merchant
                            }
                        })
            except Exception as inner_ex:
                logger.warning(f"Failed to process history for session {s.id}: {inner_ex}")
                continue
                
        # Sort pending list by timestamp descending (newest first)
        pending_list.sort(key=lambda x: x["timestamp"], reverse=True)
        return pending_list
        
    except Exception as e:
        import traceback
        logger.error(f"Error fetching pending approvals: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/action/{session_id}")
async def resume_session(session_id: str, payload: dict):
    approved = payload.get("approved", True)
    interrupt_id = payload.get("interrupt_id", "manager_review")
    
    # Construct exact resume payload as specified
    resume_message = {
        "role": "user",
        "parts": [{
            "function_response": {
                "id": interrupt_id,
                "name": "adk_request_input",
                "response": {
                    "approved": approved
                }
            }
        }]
    }
    
    try:
        _, agent = get_clients()
        logger.info(f"Resuming session {session_id} with approved={approved}, user_id='default-user'")
        
        input_params = {
            "message": resume_message,
            "user_id": "default-user",
            "session_id": session_id
        }
        
        response_stream = agent.execution_api_client.stream_query_reasoning_engine(
            request=aip_types.StreamQueryReasoningEngineRequest(
                name=agent.resource_name,
                input=input_params,
                class_method="stream_query",
            )
        )
        
        final_text = ""
        events = []
        for chunk in response_stream:
            for parsed_json in _utils.yield_parsed_json(chunk):
                if parsed_json is not None:
                    events.append(parsed_json)
                    content = parsed_json.get("content", {})
                    if content:
                        parts = content.get("parts", [])
                        for p in parts:
                            if "text" in p:
                                final_text += p["text"]
                                
        # Fallback to output field in case content is absent
        if not final_text:
            for ev in reversed(events):
                if "output" in ev and ev["output"]:
                    final_text = ev["output"]
                    break
                    
        logger.info(f"Session {session_id} resumed. Final Response: '{final_text}'")
        return {
            "status": "success",
            "final_text": final_text
        }
        
    except Exception as e:
        import traceback
        logger.error(f"Error resuming session {session_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Expense Agent - Manager Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0c16;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.15);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.2);
            --danger: #ef4444;
            --danger-glow: rgba(239, 68, 68, 0.2);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --card-bg: rgba(255, 255, 255, 0.03);
            --card-border: rgba(255, 255, 255, 0.07);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }

        /* Ambient Glows */
        .ambient-glow-1 {
            position: absolute;
            top: -200px;
            left: -200px;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, rgba(99, 102, 241, 0) 70%);
            pointer-events: none;
            z-index: 0;
        }

        .ambient-glow-2 {
            position: absolute;
            bottom: -200px;
            right: -200px;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0) 70%);
            pointer-events: none;
            z-index: 0;
        }

        .wrapper {
            position: relative;
            z-index: 10;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2.5rem 1.5rem;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 1.5rem;
        }

        .logo-section h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .logo-section p {
            font-size: 0.95rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .refresh-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 0.75rem 1.25rem;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .refresh-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        /* Dashboard Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 2rem;
        }

        /* Sleek Glassmorphic Card */
        .expense-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.75rem;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        }

        .expense-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(130deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0) 100%);
            pointer-events: none;
            z-index: 1;
        }

        .expense-card:hover {
            transform: translateY(-6px);
            border-color: rgba(99, 102, 241, 0.35);
            box-shadow: 0 12px 40px var(--primary-glow);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1.25rem;
        }

        .merchant-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #ffffff;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 180px;
        }

        .merchant-title.empty {
            color: var(--text-muted);
            font-style: italic;
        }

        .amount-tag {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--success);
            text-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
        }

        .expense-desc {
            font-size: 1rem;
            color: var(--text-main);
            margin-bottom: 1rem;
            line-height: 1.4;
        }

        .warning-box {
            background: rgba(239, 68, 68, 0.06);
            border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 12px;
            padding: 0.85rem;
            font-size: 0.85rem;
            color: #fca5a5;
            margin-bottom: 1.5rem;
            line-height: 1.4;
        }

        .meta-info {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 1.5rem;
        }

        .card-actions {
            display: flex;
            gap: 1rem;
            z-index: 2;
        }

        .btn {
            flex: 1;
            padding: 0.85rem;
            border-radius: 12px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
            border: none;
            outline: none;
        }

        .btn-approve {
            background: var(--success);
            color: #ffffff;
            box-shadow: 0 4px 15px var(--success-glow);
        }

        .btn-approve:hover:not(:disabled) {
            background: #059669;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }

        .btn-reject {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #fca5a5;
        }

        .btn-reject:hover:not(:disabled) {
            background: var(--danger);
            color: #ffffff;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px var(--danger-glow);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }

        /* Loading Spinner */
        .spinner {
            width: 18px;
            height: 18px;
            border: 2.5px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #ffffff;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* No Claims State */
        .empty-state {
            grid-column: 1 / -1;
            text-align: center;
            padding: 5rem 2rem;
            background: var(--card-bg);
            border: 1px dashed var(--card-border);
            border-radius: 24px;
            backdrop-filter: blur(12px);
        }

        .empty-state h3 {
            font-size: 1.5rem;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }

        .empty-state p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        /* Sliding Panel / Modal */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(11, 12, 22, 0.6);
            backdrop-filter: blur(8px);
            z-index: 100;
            opacity: 0;
            visibility: hidden;
            transition: all 0.4s ease;
        }

        .modal-overlay.active {
            opacity: 1;
            visibility: visible;
        }

        .modal-sidebar {
            position: fixed;
            top: 0;
            right: 0;
            width: 100%;
            max-width: 500px;
            height: 100vh;
            background: rgba(15, 17, 32, 0.92);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-left: 1px solid var(--card-border);
            z-index: 110;
            transform: translateX(100%);
            transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            padding: 3rem 2rem;
            display: flex;
            flex-direction: column;
            box-shadow: -10px 0 40px rgba(0, 0, 0, 0.5);
        }

        .modal-overlay.active .modal-sidebar {
            transform: translateX(0);
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }

        .modal-header h2 {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .close-btn {
            background: rgba(255, 255, 255, 0.05);
            border: none;
            color: var(--text-main);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 1.2rem;
            transition: all 0.2s ease;
        }

        .close-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: scale(1.1);
        }

        .modal-body {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .section-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .meta-detail-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.25rem;
        }

        .meta-detail-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .detail-item label {
            font-size: 0.8rem;
            color: var(--text-muted);
            display: block;
            margin-bottom: 0.25rem;
        }

        .detail-item span {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
        }

        .compliance-text {
            background: rgba(99, 102, 241, 0.04);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 16px;
            padding: 1.5rem;
            font-size: 1rem;
            line-height: 1.6;
            color: #e0e7ff;
            white-space: pre-line;
            box-shadow: inset 0 0 20px rgba(99, 102, 241, 0.05);
        }

        /* Success/Failure Glow Accents on Modal */
        .modal-sidebar.approved-accent {
            border-left: 2px solid var(--success);
            box-shadow: -10px 0 40px rgba(16, 185, 129, 0.1);
        }
        .modal-sidebar.rejected-accent {
            border-left: 2px solid var(--danger);
            box-shadow: -10px 0 40px rgba(239, 68, 68, 0.1);
        }
    </style>
</head>
<body>
    <div class="ambient-glow-1"></div>
    <div class="ambient-glow-2"></div>

    <div class="wrapper">
        <header>
            <div class="logo-section">
                <h1>Expense Manager Portal</h1>
                <p>Human-in-the-Loop approval dashboard powered by Gemini</p>
            </div>
            <button class="refresh-btn" onclick="fetchPendingClaims()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                Sync Claims
            </button>
        </header>

        <div class="dashboard-grid" id="claimsGrid">
            <!-- Loading indicator -->
            <div class="empty-state">
                <h3>Scanning reasoning engines...</h3>
                <p>Checking for pending manager approval interrupts.</p>
            </div>
        </div>
    </div>

    <!-- Slide out Compliance Modal -->
    <div class="modal-overlay" id="modalOverlay" onclick="closeModal()">
        <div class="modal-sidebar" id="modalSidebar" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h2 id="modalTitle">Compliance Review</h2>
                <button class="close-btn" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div>
                    <div class="section-label">Claim Details</div>
                    <div class="meta-detail-box">
                        <div class="meta-detail-grid">
                            <div class="detail-item">
                                <label>Merchant</label>
                                <span id="modalMerchant">-</span>
                            </div>
                            <div class="detail-item">
                                <label>Amount</label>
                                <span id="modalAmount">-</span>
                            </div>
                            <div class="detail-item" style="grid-column: 1 / -1; margin-top: 0.5rem;">
                                <label>Description</label>
                                <span id="modalDesc">-</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div>
                    <div class="section-label">Agent Output & Compliance Check</div>
                    <div class="compliance-text" id="modalComplianceText">
                        Awaiting final decision processing...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function fetchPendingClaims() {
            const grid = document.getElementById('claimsGrid');
            grid.innerHTML = `
                <div class="empty-state">
                    <h3>Syncing with Vertex AI Agent Runtime...</h3>
                    <p>Fetching active sessions and checking logs.</p>
                </div>
            `;

            try {
                const response = await fetch('/api/pending');
                const claims = await response.json();

                if (!Array.isArray(claims) || claims.length === 0) {
                    grid.innerHTML = `
                        <div class="empty-state">
                            <h3>All Clear! 🎉</h3>
                            <p>No claims require manager review at this time.</p>
                        </div>
                    `;
                    return;
                }

                grid.innerHTML = '';
                claims.forEach(claim => {
                    const card = document.createElement('div');
                    card.className = 'expense-card';
                    card.id = `card-${claim.session_id}`;
                    
                    const merchantName = claim.expense.merchant ? claim.expense.merchant : 'General Expense';
                    const formattedAmount = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(claim.expense.amount);
                    const formattedDate = new Date(claim.timestamp * 1000).toLocaleString();

                    card.innerHTML = `
                        <div>
                            <div class="card-header">
                                <span class="merchant-title ${!claim.expense.merchant ? 'empty' : ''}">${merchantName}</span>
                                <span class="amount-tag">${formattedAmount}</span>
                            </div>
                            <div class="expense-desc">${claim.expense.description}</div>
                            <div class="warning-box">${claim.message}</div>
                        </div>
                        <div>
                            <div class="meta-info">
                                <span>Session: ${claim.session_id.substring(0, 12)}...</span>
                                <span>${formattedDate}</span>
                            </div>
                            <div class="card-actions">
                                <button class="btn btn-reject" onclick="processAction('${claim.session_id}', '${claim.interrupt_id}', false, this)">
                                    <div class="spinner"></div>
                                    <span class="btn-text">Reject</span>
                                </button>
                                <button class="btn btn-approve" onclick="processAction('${claim.session_id}', '${claim.interrupt_id}', true, this)">
                                    <div class="spinner"></div>
                                    <span class="btn-text">Approve</span>
                                </button>
                            </div>
                        </div>
                    `;
                    grid.appendChild(card);
                });

            } catch (error) {
                console.error('Failed to fetch pending claims:', error);
                grid.innerHTML = `
                    <div class="empty-state">
                        <h3 style="color: var(--danger);">System Synchronization Error</h3>
                        <p>Could not connect to Reasoning Engine: ${error.message}</p>
                    </div>
                `;
            }
        }

        async function processAction(sessionId, interruptId, approved, buttonElement) {
            const card = document.getElementById(`card-${sessionId}`);
            const actions = card.querySelectorAll('.btn');
            
            // Disable all buttons in this card and show spinner
            actions.forEach(btn => btn.disabled = true);
            const spinner = buttonElement.querySelector('.spinner');
            const btnText = buttonElement.querySelector('.btn-text');
            spinner.style.display = 'block';
            btnText.style.display = 'none';

            try {
                const response = await fetch(`/api/action/${sessionId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ approved, interrupt_id: interruptId })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    // Populate modal contents
                    const claimCard = document.getElementById(`card-${sessionId}`);
                    document.getElementById('modalMerchant').innerText = claimCard.querySelector('.merchant-title').innerText;
                    document.getElementById('modalAmount').innerText = claimCard.querySelector('.amount-tag').innerText;
                    document.getElementById('modalDesc').innerText = claimCard.querySelector('.expense-desc').innerText;
                    
                    const complianceText = document.getElementById('modalComplianceText');
                    complianceText.innerText = result.final_text;

                    // Set visual accent for decision
                    const sidebar = document.getElementById('modalSidebar');
                    sidebar.className = 'modal-sidebar ' + (approved ? 'approved-accent' : 'rejected-accent');
                    document.getElementById('modalTitle').innerText = approved ? 'Approved Compliance Audit' : 'Rejected Compliance Audit';

                    // Show modal
                    document.getElementById('modalOverlay').classList.add('active');

                    // Fade card and remove it
                    card.style.opacity = '0';
                    card.style.transform = 'scale(0.9)';
                    setTimeout(() => {
                        card.remove();
                        // If no cards left, show empty state
                        const grid = document.getElementById('claimsGrid');
                        if (grid.children.length === 0) {
                            grid.innerHTML = `
                                <div class="empty-state">
                                    <h3>All Clear! 🎉</h3>
                                    <p>No claims require manager review at this time.</p>
                                </div>
                            `;
                        }
                    }, 400);

                } else {
                    alert('Failed to process claim: ' + (result.message || 'Unknown error'));
                    // Reset buttons
                    actions.forEach(btn => btn.disabled = false);
                    spinner.style.display = 'none';
                    btnText.style.display = 'block';
                }

            } catch (error) {
                console.error('Error processing manager action:', error);
                alert('Connection failure processing action: ' + error.message);
                // Reset buttons
                actions.forEach(btn => btn.disabled = false);
                spinner.style.display = 'none';
                btnText.style.display = 'block';
            }
        }

        function closeModal() {
            document.getElementById('modalOverlay').classList.remove('active');
        }

        // Initial scan
        window.addEventListener('DOMContentLoaded', fetchPendingClaims);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
