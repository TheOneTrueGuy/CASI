from flask import render_template, redirect, url_for, flash, request, abort, jsonify, current_app, session, make_response
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi, BaseView, expose, has_access
from flask_appbuilder import SimpleFormView
from flask_appbuilder.forms import DynamicForm
from flask_login import current_user
from . import appbuilder, db

from .models import Individual, ListGroup, Subscription, SUBSCRIPTION_TIERS
from .forms import PE_Form, waitForm
from wtforms import StringField, DecimalField
from wtforms.validators import DataRequired
from collections import deque
import openai
import logging
import json
import os
import uuid  # Added for trace ID generation

from .payment_manager import PaymentManager
from .runpod_manager import RunPodManager
from . import webCASI as casi

# Securely get OpenAI API key
openai_api_key_env = os.environ.get('OPENAI_API_KEY')
if not openai_api_key_env:
    logging.warning("OPENAI_API_KEY environment variable not set. OpenAI features will not work.")
    openai.api_key = None
else:
    openai.api_key = openai_api_key_env

stripe_secret_key_env = os.environ.get('STRIPE_SECRET_KEY')
runpod_api_key_env = os.environ.get('RUNPOD_API_KEY')

# -----------------------------
# Admin: Credit Top-Up View
# -----------------------------

class AdminTopUpForm(DynamicForm):
    username = StringField('Username', validators=[DataRequired()])
    amount = DecimalField('Amount', validators=[DataRequired()])

class AdminCreditTopUpView(SimpleFormView):
    form = AdminTopUpForm
    form_title = 'Admin Credit Top-Up'
    message = 'Credits updated.'

    @has_access
    def form_get(self, form):
        pass

    @has_access
    def form_post(self, form):
        roles = [r.name for r in getattr(current_user, 'roles', [])]
        if 'Admin' not in roles:
            abort(403)
        username = form.username.data.strip()
        amount = float(form.amount.data)
        from .models import Individual
        user = db.session.query(Individual).filter_by(username=username).first()
        if not user:
            flash(f"User '{username}' not found.", 'danger')
            return
        try:
            payment_manager.add_credits(
                user_id=user.id,
                amount=amount,
                transaction_type='admin_adjust',
                description=f'Admin top-up for {username}'
            )
            db.session.commit()
            flash(self.message + f" New balance: {user.credits_balance}", 'info')
        except Exception as e:
            db.session.rollback()
            flash(f"Error updating credits: {e}", 'danger')

appbuilder.add_view(
    AdminCreditTopUpView,
    "Credit Top-Up",
    icon="fa-plus-circle",
    category="Billing",
    category_icon="fa-money",
)

if not stripe_secret_key_env:
    logging.warning("STRIPE_SECRET_KEY environment variable not set. Payment features might not work correctly.")
if not runpod_api_key_env:
    logging.warning("RUNPOD_API_KEY environment variable not set. RunPod features might not work correctly.")

payment_manager = PaymentManager(stripe_secret_key=stripe_secret_key_env)
if hasattr(db, 'session'):
    payment_manager.set_db_session(db.session)
else:
    logging.error("Database session not available.")

runpod_manager = RunPodManager(api_key=runpod_api_key_env, payment_manager=payment_manager)

# Helper classes
sendcount=0
class Stack:
    def __init__(self):
        self.stack = deque(maxlen=9)
        self.count = 0
    def push(self, string1, string2, string3):
        self.stack.append([string1, string2, string3])
        self.count += 1
        if self.count>19: self.count=19
    def get_stack(self):
        return list(self.stack)
    def get_item(self, index):
        if index >= 0 and index < len(self.stack):
          return self.stack[index]
        else:
          return ["error", "error", "error"]

stacky=Stack()

class MessageBuilder:
    def __init__(self):
        self.msg = []
    def add_line(self, role, content):
        line = {"role": role, "content": content}
        self.msg.append(line)
    def get_message(self):
        return self.msg
    def clear(self):
        self.msg.clear()

builder = MessageBuilder()

def gen4(prompt, temp=0.0):
    response = openai.ChatCompletion.create(model="gpt-4o-2024-08-06",messages=prompt, temperature=temp)
    response_dict = dict(response)
    reply = response_dict['choices'][0]['message']['content']
    return reply

# ContactModelView and GroupModelView removed to fix KeyError: 'gender'
# They were commented out in previous working versions.

class waitFormView(SimpleFormView):
    form = waitForm
    form_title = 'Waitlist Form'
    message = 'Thanks for your submission!'
    def form_get(self, form):
        form.field_string.data = 'Default Value'
    def form_post(self, form):
        flash(self.message, 'info')
        
appbuilder.add_view(waitFormView, "Wait Form", icon = "fa-group", label=("Waitlist Form"), category = "Forms", category_icon = "fa-cogs")

class PEFormView(SimpleFormView):
    form = PE_Form
    form_title = 'Prompt Engine Form'
    message = 'Thanks for your submission!'
    def form_get(self, form):
        form.text1.data = ""
        form.text2.data = ""
        form.text3.data = ""
    def form_post(self, form):
        global sendcount
        sendcount+=1
        if sendcount > 19:
            sendcount=0
            stacky.stack.clear()
        sys = form.text1.data
        user = form.text2.data
        builder.clear()
        builder.add_line("system", sys)
        builder.add_line("user", user)
        msg=builder.get_message()
        reply=gen4(msg)
        form.text3.data = reply
        stacky.push(sys, user, reply)
        flash(self.message, 'info')

appbuilder.add_view(PEFormView, "PE form View", icon="fa-group", label=("PE forms"), category="Prompt Engine", category_icon="fa-cogs")

class Admin(BaseView):
    default_view = 'admin'
    @has_access
    @expose('/dashboard/')
    def admin(self):
        return self.render_template('admin.html')

appbuilder.add_view(Admin, "Dashboard", icon="fa-dashboard", category="Admin", category_icon="fa-cogs")

from flask_login import current_user

class BillingProfileView(BaseView):
    route_base = "/billing"
    default_view = "profile"

    @expose('/profile/')
    @has_access
    def profile(self):
        if not current_user.is_authenticated:
            flash("Please log in to view your billing profile.", "warning")
            return redirect(appbuilder.get_url_for_login)

        user_subscription = db.session.query(Subscription).filter_by(user_id=current_user.id).order_by(Subscription.id.desc()).first()
        user_profile = db.session.query(Individual).filter_by(id=current_user.id).first()
        credit_balance = user_profile.credits_balance if user_profile else 0
        credit_packages = current_app.config.get('CREDIT_PACKAGES', {})
        
        self.update_redirect()
        return self.render_template(
            'billing_profile.html',
            user_subscription=user_subscription,
            credit_balance=credit_balance,
            subscription_tiers=SUBSCRIPTION_TIERS,
            credit_packages=credit_packages
        )

    @expose('/subscribe/<string:tier_id>')
    @has_access
    def subscribe(self, tier_id):
        if not current_user.is_authenticated:
            flash("Please log in to subscribe.", "warning")
            return redirect(appbuilder.get_url_for_login)

        if not payment_manager.stripe_secret_key:
            flash("Stripe payments are not configured.", "danger")
            return redirect(self.get_redirect())

        tier_info = SUBSCRIPTION_TIERS.get(tier_id)
        if not tier_info:
            flash("Invalid subscription tier selected.", "danger")
            return redirect(self.get_redirect())

        stripe_price_id = tier_info.get('stripe_price_id')
        if not stripe_price_id or 'replace_me' in stripe_price_id:
            flash(f"Payment config incomplete for '{tier_info['name']}'.", "warning")
            return redirect(self.get_redirect())
            
        success_url = url_for('.profile', _external=True) + '?session_id={CHECKOUT_SESSION_ID}&status=success_subscription'
        cancel_url = url_for('.profile', _external=True) + '?status=cancel_subscription'

        try:
            checkout_session_url = payment_manager.create_subscription_checkout_session(
                user_id=current_user.id,
                user_email=current_user.email,
                stripe_price_id=stripe_price_id,
                success_url=success_url,
                cancel_url=cancel_url
            )
            if checkout_session_url:
                return redirect(checkout_session_url)
            else:
                flash("Could not initiate Stripe Checkout.", "danger")
        except Exception as e:
            flash(f"Error setting up subscription: {str(e)}", "danger")
        
        return redirect(self.get_redirect())

    @expose('/purchase_credits/<string:package_id>')
    @has_access
    def purchase_credits(self, package_id):
        if not current_user.is_authenticated:
            flash("Please log in to purchase credits.", "warning")
            return redirect(appbuilder.get_url_for_login)

        if not payment_manager.stripe_secret_key:
            flash("Stripe payments are not configured.", "danger")
            return redirect(self.get_redirect())

        credit_packages = current_app.config.get('CREDIT_PACKAGES', {})
        package_info = credit_packages.get(package_id)

        if not package_info:
            flash("Invalid credit package.", "danger")
            return redirect(self.get_redirect())

        stripe_price_id = package_info.get('stripe_price_id')
        if not stripe_price_id or 'replace_me' in stripe_price_id:
            flash(f"Payment config incomplete for '{package_info['name']}'.", "warning")
            return redirect(self.get_redirect())

        success_url = url_for('.profile', _external=True) + '?session_id={CHECKOUT_SESSION_ID}&status=success_credits'
        cancel_url = url_for('.profile', _external=True) + '?status=cancel_credits'
        
        try:
            checkout_session_url = payment_manager.create_one_time_checkout_session(
                user_id=current_user.id,
                user_email=current_user.email,
                package_name=package_info['name'],
                package_credits=package_info['credits'],
                stripe_price_id=stripe_price_id,
                success_url=success_url,
                cancel_url=cancel_url
            )
            if checkout_session_url:
                return redirect(checkout_session_url)
            else:
                flash("Could not initiate Stripe Checkout.", "danger")
        except Exception as e:
            flash(f"Error purchasing credits: {str(e)}", "danger")

        return redirect(self.get_redirect())

appbuilder.add_view(BillingProfileView, "Billing Profile", icon="fa-credit-card", label=("Billing"), category="Profile", category_icon="fa-user")

@appbuilder.app.route('/stripe-webhooks', methods=['POST'])
def stripe_webhook():
    webhook_secret = current_app.config.get('STRIPE_WEBHOOK_SECRET') 
    if not webhook_secret:
        return jsonify(status="error", message="Webhook secret not configured."), 200

    payload_string = request.data.decode('utf-8')
    sig_header = request.headers.get('Stripe-Signature')

    if not payload_string or not sig_header:
        return jsonify(error="Missing payload or signature"), 400

    success, message = payment_manager.handle_webhook_event(payload_string, sig_header, webhook_secret)

    if success:
        return jsonify(status="success", message=message), 200
    else:
        return jsonify(error=message), 400

class VideoOverlayView(BaseView):
    route_base = "/video_overlay"
    default_view = "show"

    @expose('/')
    @has_access
    def show(self):
        return self.render_template('video_overlay.html')

appbuilder.add_view(VideoOverlayView, "Video Overlay Tool", icon="fa-video-camera", label="Video Overlay", category="Tools", category_icon="fa-wrench")

# --- CASI Tool with File-Based History Storage ---
class CasiView(BaseView):
    route_base = "/casi"
    default_view = "tool"

    @expose('/', methods=['GET', 'POST'])
    def tool(self):
        if 'casi_thread' not in session:
            session['casi_thread'] = []
        # Ensure API keys in session
        for key in ['openai_api_key', 'anthropic_api_key', 'openrouter_api_key']:
            if key not in session: session[key] = ''

        # Clean up legacy large history if present to free up cookie space
        if 'casi_history' in session:
            session.pop('casi_history', None)
            session.modified = True

        prompts = {
            "generator": "Formalize and expand this idea.",
            "critic": "Analyze and critique this idea."
        }
        available_backends = ["openai", "anthropic", "google", "openrouter"]
        
        context = {
            "generator_prompt": prompts["generator"],
            "critic_prompt": prompts["critic"],
            "generator_input": "",
            "generator_output": "",
            "critic_input": "",
            "critic_output": "",
            "backends": available_backends,
            "selected_gen_backend": "openrouter",
            "selected_crit_backend": "openrouter",
            # Renamed keys to break browser autofill
            "gen_model_id": "qwen/qwen3-32b",
            "crit_model_id": "qwen/qwen3-32b",
        }

        # Check for history ID in session
        trace_id = session.get('casi_trace_id')
        has_history = False
        if trace_id:
            # Verify file exists
            if os.path.exists(f"/tmp/casi_trace_{trace_id}.json"):
                has_history = True
        
        context['has_history'] = has_history

        def load_history_from_session():
            trace_id = session.get('casi_trace_id')
            if trace_id and os.path.exists(f"/tmp/casi_trace_{trace_id}.json"):
                try:
                    with open(f"/tmp/casi_trace_{trace_id}.json", "r") as f:
                        return json.load(f)
                except:
                    return []
            return []

        def save_history_to_session(history):
            trace_id = session.get('casi_trace_id')
            if not trace_id:
                trace_id = str(uuid.uuid4())
                session['casi_trace_id'] = trace_id
            
            with open(f"/tmp/casi_trace_{trace_id}.json", "w") as f:
                json.dump(history, f)
            return trace_id

        if request.method == 'POST':
            context['selected_gen_backend'] = request.form.get('generator_backend', 'openrouter')
            context['selected_crit_backend'] = request.form.get('critic_backend', 'openrouter')
            context['gen_model_id'] = request.form.get('gen_model_id') or context.get('gen_model_id')
            context['crit_model_id'] = request.form.get('crit_model_id') or context.get('crit_model_id')
            context['generator_prompt'] = request.form.get('generator_prompt')
            context['critic_prompt'] = request.form.get('critic_prompt')
            context['generator_input'] = request.form.get('generator_input')
            context['critic_input'] = request.form.get('critic_input', '')
            context['critic_output'] = request.form.get('critic_output', '')
            context['max_iterations'] = request.form.get('max_iterations', 5)

            action = request.form.get('action')

            if action == 'download_trace':
                if has_history:
                    try:
                        with open(f"/tmp/casi_trace_{trace_id}.json", "r") as f:
                            history = json.load(f)
                        trace_text = casi.format_history_as_text(history)
                        response = make_response(trace_text)
                        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
                        response.headers['Content-Disposition'] = 'attachment; filename="casi_trace.txt"'
                        return response
                    except Exception as e:
                        flash(f"Error retrieving trace: {e}", 'danger')
                else:
                    flash('No CASI history available to download.', 'warning')

            if action == 'save_keys':
                session['openai_api_key'] = request.form.get('openai_api_key', '')
                session['anthropic_api_key'] = request.form.get('anthropic_api_key', '')
                session['openrouter_api_key'] = request.form.get('openrouter_api_key', '')
                flash('API keys updated.', 'info')

            elif action == 'run_generator':
                # Load existing history
                history = load_history_from_session()
                
                api_key = None
                if context['selected_gen_backend'] == 'openai': api_key = session.get('openai_api_key')
                elif context['selected_gen_backend'] == 'anthropic': api_key = session.get('anthropic_api_key')
                elif context['selected_gen_backend'] == 'openrouter': api_key = session.get('openrouter_api_key')

                # Determine model to use (override or default)
                gen_model = context.get('gen_model_id')
                if not gen_model:
                    gen_model = getattr(casi.config, f"{context['selected_gen_backend']}_model", None)
                    if context['selected_gen_backend'] == 'openrouter' and (not gen_model or 'deepseek' in gen_model):
                        gen_model = 'qwen/qwen3-32b'

                # Prepare Input with Context from History
                initial_input = session.get('casi_initial_input', context['generator_input']) # Use session or current input
                if not session.get('casi_initial_input'): session['casi_initial_input'] = initial_input # Save if new
                
                critic_feedback = session.get('casi_last_critic_feedback', context['critic_output'])
                history_text = casi.format_history_as_text(history)
                
                # Construct prompt/input
                current_gen_prompt = context['generator_prompt']
                # Note: We don't auto-switch prompt in manual mode unless user asks, but we could.
                # For consistency, we'll use the prompt exactly as in the text box.
                
                # If history exists, we should augment the input
                gen_input_text = context['generator_input']
                if history:
                     gen_input_text = f"ORIGINAL GOAL: {initial_input}\n\nPREVIOUS HISTORY:\n{history_text}\n\nLATEST CRITIQUE:\n{critic_feedback}\n\nCURRENT TASK:\n{context['generator_input']}"

                gen_output, _, gen_trace = casi.generator(
                    backend=context['selected_gen_backend'],
                    model=gen_model,
                    prompt=current_gen_prompt,
                    user_input=gen_input_text,
                    critic_feedback="", # Context embedded
                    api_key=api_key
                )
                
                context['generator_output'] = gen_output
                context['critic_input'] = gen_output
                
                # Update History
                # If last item is incomplete (no critic output), update it. Else append new.
                if history and not history[-1].get('critic_output'):
                    history[-1]['generator_output'] = gen_output
                    history[-1]['generator_trace'] = gen_trace
                else:
                    history.append({
                        'iteration': len(history) + 1,
                        'generator_input': context['generator_input'],
                        'critic_feedback_input': critic_feedback,
                        'generator_output': gen_output,
                        'critic_output': "",
                        'generator_trace': gen_trace,
                        'critic_trace': {}
                    })
                save_history_to_session(history)
                
                # Update Session State for consistency with Auto Mode
                session['casi_last_gen_output'] = gen_output
                session['casi_last_gen_trace'] = gen_trace
                session['casi_auto_next'] = 'critic' # Next logical step is critic

            elif action == 'run_critic':
                # Load existing history
                history = load_history_from_session()

                api_key = None
                if context['selected_crit_backend'] == 'openai': api_key = session.get('openai_api_key')
                elif context['selected_crit_backend'] == 'anthropic': api_key = session.get('anthropic_api_key')
                elif context['selected_crit_backend'] == 'openrouter': api_key = session.get('openrouter_api_key')

                # Determine model to use (override or default)
                crit_model = context.get('crit_model_id')
                if not crit_model: 
                    crit_model = getattr(casi.config, f"{context['selected_crit_backend']}_model", None)
                    if context['selected_crit_backend'] == 'openrouter' and (not crit_model or 'deepseek' in crit_model):
                        crit_model = 'qwen/qwen3-32b'

                crit_output, _, crit_trace = casi.critic(
                    backend=context['selected_crit_backend'],
                    model=crit_model,
                    prompt=context['critic_prompt'],
                    generator_output=context['critic_input'],
                    api_key=api_key
                )
                context['critic_output'] = crit_output
                
                # Update History
                # If last item is incomplete, update it. Else append new (weird but possible).
                if history and not history[-1].get('critic_output'):
                    history[-1]['critic_output'] = crit_output
                    history[-1]['critic_trace'] = crit_trace
                else:
                    # Orphaned critique
                    history.append({
                        'iteration': len(history) + 1,
                        'generator_input': "(Manual Critique Only)",
                        'critic_feedback_input': "",
                        'generator_output': context['critic_input'],
                        'critic_output': crit_output,
                        'generator_trace': {},
                        'critic_trace': crit_trace
                    })
                save_history_to_session(history)
                
                # Update Session State
                session['casi_last_critic_feedback'] = crit_output
                session['casi_auto_next'] = 'generator' # Next logical step is generator
                
                # UX Improvement: Automatically switch Generator prompt to "Iteration Mode"
                current_gen_prompt = context.get('generator_prompt', '').strip()
                initial_gen_prompt = casi.config.prompts.get("generator_initial", "").strip()
                iter_gen_prompt = casi.config.prompts.get("generator_iteration", "").strip()
                
                if current_gen_prompt == initial_gen_prompt and iter_gen_prompt:
                    context['generator_prompt'] = iter_gen_prompt
                    flash("Generator prompt updated to 'Iteration Mode' for the next turn.", "info")

            elif action == 'run_cycle' or action == 'step_cycle':
                # Common parameter setup
                try:
                    max_iterations = int(request.form.get('max_iterations', 5))
                    if not (1 <= max_iterations <= 20): max_iterations = 5
                except (ValueError, TypeError): max_iterations = 5
                
                # ... (Key/Model setup same as before, handled by shared code below if I merge)
                # Since I am replacing the block, I need to keep the key/model logic here or deduplicate.
                # I will keep it inline for safety.
                
                # Get keys
                gen_api_key = None
                if context['selected_gen_backend'] == 'openai': gen_api_key = session.get('openai_api_key')
                elif context['selected_gen_backend'] == 'anthropic': gen_api_key = session.get('anthropic_api_key')
                elif context['selected_gen_backend'] == 'openrouter': gen_api_key = session.get('openrouter_api_key')

                crit_api_key = None
                if context['selected_crit_backend'] == 'openai': crit_api_key = session.get('openai_api_key')
                elif context['selected_crit_backend'] == 'anthropic': crit_api_key = session.get('anthropic_api_key')
                elif context['selected_crit_backend'] == 'openrouter': crit_api_key = session.get('openrouter_api_key')

                # Get models
                gen_model = context.get('gen_model_id')
                if not gen_model: 
                    gen_model = getattr(casi.config, f"{context['selected_gen_backend']}_model", None)
                    if context['selected_gen_backend'] == 'openrouter' and (not gen_model or 'deepseek' in gen_model):
                        gen_model = 'qwen/qwen3-32b'
                
                crit_model = context.get('crit_model_id')
                if not crit_model: 
                    crit_model = getattr(casi.config, f"{context['selected_crit_backend']}_model", None)
                    if context['selected_crit_backend'] == 'openrouter' and (not crit_model or 'deepseek' in crit_model):
                        crit_model = 'qwen/qwen3-32b'

                # --- Cycle Logic ---
                history = load_history_from_session()
                
                if action == 'run_cycle':
                    # START or RESUME
                    # If history exists, we resume. If empty, we start fresh.
                    
                    if not history:
                        # Fresh Start
                        new_trace_id = str(uuid.uuid4())
                        session['casi_trace_id'] = new_trace_id
                        context['has_history'] = True
                        history = []
                        save_history_to_session(history)
                        
                        session['casi_auto_iter'] = 1
                        session['casi_auto_next'] = 'critic' # Will run Generator below, so next is critic
                        session['casi_initial_input'] = context['generator_input']
                        session['casi_last_critic_feedback'] = ""
                        
                        flash(f"Automatic Cycle Started (Round 1/{max_iterations}). Generator is thinking...", "info")
                        gen_output, _, gen_trace = casi.generator(
                            backend=context['selected_gen_backend'], 
                            model=gen_model, 
                            prompt=context['generator_prompt'],
                            user_input=context['generator_input'], 
                            critic_feedback="", 
                            api_key=gen_api_key
                        )
                        
                        context['generator_output'] = gen_output
                        context['critic_input'] = gen_output
                        session['casi_last_gen_output'] = gen_output
                        session['casi_last_gen_trace'] = gen_trace
                        
                        session['casi_auto_active'] = True
                        session['casi_auto_max'] = max_iterations
                        context['auto_continue'] = True
                        
                    else:
                        # Resume from History
                        last_item = history[-1]
                        current_iter = last_item['iteration']
                        
                        if not last_item.get('critic_output'):
                            # Incomplete iteration (Generator ran, Critic pending)
                            session['casi_auto_next'] = 'critic'
                            session['casi_auto_iter'] = current_iter
                            session['casi_last_gen_output'] = last_item.get('generator_output', '')
                            session['casi_last_gen_trace'] = last_item.get('generator_trace', {})
                            # Ensure context is synced
                            context['generator_output'] = last_item.get('generator_output', '')
                            context['critic_input'] = last_item.get('generator_output', '')
                            
                        else:
                            # Complete iteration (Critic ran, Generator pending for next)
                            session['casi_auto_next'] = 'generator'
                            session['casi_auto_iter'] = current_iter + 1
                            session['casi_last_critic_feedback'] = last_item.get('critic_output', '')
                            session['casi_initial_input'] = history[0].get('generator_input', '') # Try to recover initial
                        
                        # Resume
                        session['casi_auto_active'] = True
                        # If resuming, ensure max_iterations is at least current + user request, or just user request as target?
                        # Let's treat input as "Target Max". If current is 3 and target is 5, we do 2 more.
                        # If target <= current, we just stop (or do 1?). Let's do at least 1 step if clicked.
                        if max_iterations <= session['casi_auto_iter']:
                             max_iterations = session['casi_auto_iter'] + 1
                             
                        session['casi_auto_max'] = max_iterations
                        
                        flash(f"Resuming Automatic Cycle at Round {session['casi_auto_iter']}/{max_iterations}...", "info")
                        context['auto_continue'] = True # Trigger step_cycle immediately

                elif action == 'step_cycle':
                    # CONTINUE: Execute next step based on session state
                    if not session.get('casi_auto_active'):
                        flash("Automatic cycle stopped or invalid state.", "warning")
                    else:
                        history = load_history_from_session()
                        current_iter = session.get('casi_auto_iter', 1)
                        max_iter = session.get('casi_auto_max', 5)
                        next_role = session.get('casi_auto_next', 'generator')
                        
                        if next_role == 'critic':
                            # Run Critic
                            flash(f"Round {current_iter}/{max_iter}: Critic is thinking...", "info")
                            crit_input = session.get('casi_last_gen_output', '')
                            
                            # Use iteration prompt if available
                            current_crit_prompt = context['critic_prompt']
                            if current_iter > 1:
                                iter_prompt = casi.config.prompts.get("critic_iteration")
                                if iter_prompt and current_crit_prompt == casi.config.prompts.get("critic_initial"):
                                    current_crit_prompt = iter_prompt

                            crit_output, _, crit_trace = casi.critic(
                                backend=context['selected_crit_backend'], 
                                model=crit_model, 
                                prompt=current_crit_prompt,
                                generator_output=crit_input, 
                                api_key=crit_api_key
                            )
                            
                            # Save Iteration to History
                            history.append({
                                'iteration': current_iter,
                                'generator_input': session.get('casi_initial_input') if current_iter == 1 else "(From previous critique)",
                                'critic_feedback_input': session.get('casi_last_critic_feedback', ''),
                                'generator_output': session.get('casi_last_gen_output', ''),
                                'critic_output': crit_output,
                                'generator_trace': session.get('casi_last_gen_trace', {}),
                                'critic_trace': crit_trace
                            })
                            save_history_to_session(history)
                            
                            # Update State
                            session['casi_last_critic_feedback'] = crit_output
                            context['critic_output'] = crit_output
                            context['generator_output'] = session.get('casi_last_gen_output', '') # Keep gen output visible
                            context['critic_input'] = session.get('casi_last_gen_output', '')
                            
                            # Check completion
                            if current_iter >= max_iter:
                                session['casi_auto_active'] = False
                                flash("Automatic cycle completed successfully.", "success")
                            else:
                                session['casi_auto_iter'] = current_iter + 1
                                session['casi_auto_next'] = 'generator'
                                context['auto_continue'] = True # Trigger next step
                        
                        elif next_role == 'generator':
                            # Run Generator
                            flash(f"Round {current_iter}/{max_iter}: Generator is thinking...", "info")
                            
                            # Prepare Input
                            initial_input = session.get('casi_initial_input', '')
                            critic_feedback = session.get('casi_last_critic_feedback', '')
                            history_text = casi.format_history_as_text(history)
                            
                            # Switch prompt to iteration mode
                            current_gen_prompt = context['generator_prompt']
                            iter_prompt = casi.config.prompts.get("generator_iteration")
                            if iter_prompt and current_gen_prompt == casi.config.prompts.get("generator_initial"):
                                current_gen_prompt = iter_prompt
                                context['generator_prompt'] = iter_prompt # Update display
                            
                            gen_input_text = f"ORIGINAL GOAL: {initial_input}\n\nPREVIOUS HISTORY:\n{history_text}\n\nLATEST CRITIQUE:\n{critic_feedback}"
                            context['generator_input'] = gen_input_text # Update display for context
                            
                            gen_output, _, gen_trace = casi.generator(
                                backend=context['selected_gen_backend'], 
                                model=gen_model, 
                                prompt=current_gen_prompt,
                                user_input=gen_input_text, 
                                critic_feedback="", # Context is embedded in input now
                                api_key=gen_api_key
                            )
                            
                            # Update State
                            session['casi_last_gen_output'] = gen_output
                            session['casi_last_gen_trace'] = gen_trace
                            session['casi_auto_next'] = 'critic'
                            
                            context['generator_output'] = gen_output
                            context['critic_input'] = gen_output
                            context['critic_output'] = critic_feedback # Keep previous critique visible
                            
                            context['auto_continue'] = True # Trigger next step

                context['cycle_history'] = history

        
        return self.render_template('casi.html', **context)

appbuilder.add_view(CasiView, "CASI Tool", icon="fa-exchange", label="CASI Tool", category="Tools", category_icon="fa-wrench")

class BespokeGraphView(BaseView):
    route_base = "/bespoke_graph"
    default_view = "index"
    @expose("/")
    @has_access
    def index(self):
        return self.render_template("bespoke/bespoke_graph.html")

appbuilder.add_view(BespokeGraphView, "Bespoke Automata", icon="fa-cogs", label="Bespoke Automata", category="Tools", category_icon="fa-wrench")

@appbuilder.app.errorhandler(404)
def page_not_found(e):
    return (render_template("404.html", base_template=appbuilder.base_template, appbuilder=appbuilder), 404)
