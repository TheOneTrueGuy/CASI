from flask import render_template, redirect, url_for, flash, request, abort, jsonify, current_app, session
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
    @has_access
    def tool(self):
        if 'casi_thread' not in session:
            session['casi_thread'] = []
        # Ensure API keys in session
        for key in ['openai_api_key', 'anthropic_api_key', 'openrouter_api_key']:
            if key not in session: session[key] = ''

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
            "generator_model": getattr(casi.config, "openrouter_model", None),
            "critic_model": getattr(casi.config, "openrouter_model", None),
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
            context['generator_model'] = request.form.get('generator_model') or context.get('generator_model')
            context['critic_model'] = request.form.get('critic_model') or context.get('critic_model')
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
                        response = current_app.response_class(
                            trace_text,
                            mimetype='text/plain; charset=utf-8',
                        )
                        response.headers['Content-Disposition'] = 'attachment; filename=casi_trace.txt'
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
                api_key = None
                if context['selected_gen_backend'] == 'openai': api_key = session.get('openai_api_key')
                elif context['selected_gen_backend'] == 'anthropic': api_key = session.get('anthropic_api_key')
                elif context['selected_gen_backend'] == 'openrouter': api_key = session.get('openrouter_api_key')

                gen_model = context.get('generator_model')
                if not gen_model: gen_model = getattr(casi.config, f"{context['selected_gen_backend']}_model", None)

                gen_output, _, _ = casi.generator(
                    backend=context['selected_gen_backend'],
                    model=gen_model,
                    prompt=context['generator_prompt'],
                    user_input=context['generator_input'],
                    critic_feedback=context['critic_output'],
                    api_key=api_key
                )
                context['generator_output'] = gen_output
                context['critic_input'] = gen_output

            elif action == 'run_critic':
                api_key = None
                if context['selected_crit_backend'] == 'openai': api_key = session.get('openai_api_key')
                elif context['selected_crit_backend'] == 'anthropic': api_key = session.get('anthropic_api_key')
                elif context['selected_crit_backend'] == 'openrouter': api_key = session.get('openrouter_api_key')

                crit_model = context.get('critic_model')
                if not crit_model: crit_model = getattr(casi.config, f"{context['selected_crit_backend']}_model", None)

                crit_output, _, _ = casi.critic(
                    backend=context['selected_crit_backend'],
                    model=crit_model,
                    prompt=context['critic_prompt'],
                    generator_output=context['critic_input'],
                    api_key=api_key
                )
                context['critic_output'] = crit_output

            elif action == 'run_cycle':
                try:
                    max_iterations = int(request.form.get('max_iterations', 5))
                    if not (1 <= max_iterations <= 20): max_iterations = 5
                except (ValueError, TypeError): max_iterations = 5

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
                gen_model = context.get('generator_model')
                if not gen_model: gen_model = getattr(casi.config, f"{context['selected_gen_backend']}_model", None)
                crit_model = context.get('critic_model')
                if not crit_model: crit_model = getattr(casi.config, f"{context['selected_crit_backend']}_model", None)

                # Run cycle
                results = casi.run_automatic_cycle(
                    max_iterations=max_iterations,
                    initial_input=context['generator_input'],
                    gen_backend=context['selected_gen_backend'],
                    gen_model=gen_model,
                    gen_prompt=context['generator_prompt'],
                    gen_api_key=gen_api_key,
                    crit_backend=context['selected_crit_backend'],
                    crit_model=crit_model,
                    crit_prompt=context['critic_prompt'],
                    crit_api_key=crit_api_key
                )

                # Save history to file instead of session
                history = results.get('history', [])
                new_trace_id = str(uuid.uuid4())
                try:
                    with open(f"/tmp/casi_trace_{new_trace_id}.json", "w") as f:
                        json.dump(history, f)
                    session['casi_trace_id'] = new_trace_id
                    context['has_history'] = True
                except Exception as e:
                    print(f"Error saving history trace: {e}")
                    flash("Cycle completed, but failed to save history for download.", "warning")

                context['generator_output'] = results.get('final_generator_output', '')
                context['critic_output'] = results.get('final_critic_output', '')
                context['cycle_history'] = history # For display if needed
                context['max_iterations'] = max_iterations
                flash(f'Automatic cycle completed {len(history)} iterations.', 'success')
        
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
