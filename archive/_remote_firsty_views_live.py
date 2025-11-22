from flask import render_template, redirect, url_for, flash, request, abort, jsonify, current_app, session
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi, BaseView, expose, has_access
from flask_appbuilder import SimpleFormView
from flask_appbuilder.forms import DynamicForm
from flask_login import current_user
from . import appbuilder, db

from .models import Individual, ListGroup, Subscription, SUBSCRIPTION_TIERS # Ensure Subscription and SUBSCRIPTION_TIERS are imported
from .forms import PE_Form, waitForm
from wtforms import StringField, DecimalField
from wtforms.validators import DataRequired
from collections import deque
import openai
import logging
import json
import os # Ensure os is imported if not already

# Ensure payment_manager is initialized after db and appbuilder
from .payment_manager import PaymentManager
from .runpod_manager import RunPodManager
from . import webCASI as casi

# Securely get OpenAI API key from environment variable
openai_api_key_env = os.environ.get('OPENAI_API_KEY')
if not openai_api_key_env:
    logging.warning("OPENAI_API_KEY environment variable not set. OpenAI features will not work.")
    openai.api_key = None
else:
    openai.api_key = openai_api_key_env

# Initialize Payment and RunPod Managers
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
        # Restrict to Admin role
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
    logging.error("Database session (db.session) not available or db not fully initialized when PaymentManager is configured.")

runpod_manager = RunPodManager(api_key=runpod_api_key_env, payment_manager=payment_manager)

# move all these first 2 classes to a different module eventually
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

def gen4(prompt, temp=0.0): #"gpt-3.5-turbo" gpt-4-32k gpt-4
    response = openai.ChatCompletion.create(model="gpt-4o-2024-08-06",messages=prompt, temperature=temp)
    response_dict = dict(response)
    reply = response_dict['choices'][0]['message']['content']
    return reply

#move the above classes to a different module eventually
"""
class ContactModelView(ModelView):
    datamodel = SQLAInterface(Individual)

    label_columns = {'contact_group':'Contacts Group'}
    list_columns = ['name','email']

    show_fieldsets = [
        ('Summary', {'fields': ['name', 'gender', 'contact_group']}),
        (
            'Contact Info',
            {'fields': ['address', 'birthday', 'personal_phone', 'personal_celphone'], 'expanded': False},
        ),
    ]

    add_fieldsets = [
        ('Summary', {'fields': ['name', 'gender', 'contact_group']}),
        (
            'Contact Info',
            {'fields': ['address', 'birthday', 'personal_phone', 'personal_celphone'], 'expanded': False},
        ),
    ]

    edit_fieldsets = [
        ('Summary', {'fields': ['name', 'gender', 'contact_group']}),
        (
            'Contact Info',
            {'fields': ['address', 'birthday', 'personal_phone', 'personal_celphone'], 'expanded': False},
        ),
    ]


class GroupModelView(ModelView):
    datamodel = SQLAInterface(ListGroup)
    related_views = [ContactModelView]



def fill_gender():
    try:
        db.session.add(Gender(name="Male"))
        db.session.add(Gender(name="Female"))
        db.session.commit()
    except Exception:
        db.session.rollback()


appbuilder.add_view(
    GroupModelView, "List Groups", icon="fa-folder-open-o", category="Contacts", category_icon='fa-envelope'
)
appbuilder.add_view(
    ContactModelView, "List Contacts", icon="fa-envelope", category="Contacts"
)
"""






class waitFormView(SimpleFormView):
    form = waitForm
    form_title = 'Waitlist Form'
    message = 'Thanks for your submission!'

    def form_get(self, form):
        form.field_string.data = 'Default Value'

    def form_post(self, form):
        # process form
        flash(self.message, 'info')
        
appbuilder.add_view(
    waitFormView,
    "Wait Form",
    icon = "fa-group",
    label=("Waitlist Form"),
    category = "Forms",
    category_icon = "fa-cogs",
)

class PEFormView(SimpleFormView):
    form = PE_Form
    form_title = 'Prompt Engine Form'
    message = 'Thanks for your submission!'

    def form_get(self, form):
        form.text1.data = ""
        form.text2.data = ""
        form.text3.data = ""

    def form_post(self, form):
        # process form
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

"""
    flash(form.text1.data, 'info')
    flash(form.text2.data, 'info')
    flash(form.text3.data, 'info')

    sys = textbox1.get("1.0", tk.END).strip()
    user = textbox2.get("1.0", tk.END).strip()
    builder.clear()
    builder.add_line("system", sys)
    builder.add_line("user", user)
    msg=builder.get_message()

    reply=gen4(msg)

    textbox3.delete("1.0", tk.END)
    textbox3.insert(tk.END, reply)
    stacky.push(sys, user, reply)


"""




appbuilder.add_view(
    PEFormView,
    "PE form View",
    icon="fa-group",
    label=("PE forms"),
    category="Prompt Engine",
    category_icon="fa-cogs",
)

"""
class PEFormView(SimpleFormView):
    form=PEForm
    form_title="Prompt Engine Form1"
    message = "Get Cracking"
    def form_get(self, form):
        form.text1.data = " "
        form.text2.data = " "
        form.text3.data = " "
    def form_post(self, form):
        flash(self.message, "info")

appbuilder.add_view(
    PEFormView,
    "PE form View",
    icon="fa-group",
    label=("PE forms"),
    category="Prompt Engine",
    category_icon="fa-cogs",
)
"""

class Admin(BaseView):
    default_view = 'admin'

    @has_access
    @expose('/dashboard/')
    def admin(self):
        return self.render_template('admin.html')

appbuilder.add_view(
    Admin,
    "Dashboard",
    icon="fa-dashboard",
    category="Admin",
    category_icon="fa-cogs",
)
"""
# class FarStrikeView(BaseView):
#     route_base = "/farstrike"  # This defines the base URL

#     @expose("/client")  # This will map to '/farstrike/client'
#     @has_access
#     def client(self):
#         print("farstrike check")
#         return self.render_template('farstrike_client.html')

# appbuilder.add_view(
#     FarStrikeView,
#     "FarStike",
#     icon="fa-dashboard",
#     category="FarStrike",
#     category_icon="fa-cogs",
# )
"""

from flask_login import current_user # Ensure current_user is imported for BillingProfileView

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

    # Added by Cascade: Stripe Checkout routes
    @expose('/subscribe/<string:tier_id>')
    @has_access
    def subscribe(self, tier_id):
        if not current_user.is_authenticated:
            flash("Please log in to subscribe.", "warning")
            return redirect(appbuilder.get_url_for_login)

        if not payment_manager.stripe_secret_key:
            flash("Stripe payments are not configured by the administrator. Please contact support.", "danger")
            return redirect(self.get_redirect())

        tier_info = SUBSCRIPTION_TIERS.get(tier_id)
        if not tier_info:
            flash("Invalid subscription tier selected.", "danger")
            return redirect(self.get_redirect())

        stripe_price_id = tier_info.get('stripe_price_id')
        if not stripe_price_id or 'replace_me' in stripe_price_id or 'xxxxxxxxxxxxxx' in stripe_price_id:
            flash(f"The payment configuration for '{tier_info['name']}' is not yet complete. Please contact support.", "warning")
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
                flash("Could not initiate Stripe Checkout. Please try again or contact support.", "danger")
        except Exception as e:
            # In a production app, log this error: current_app.logger.error(f"Stripe subscription error for user {current_user.id}, tier {tier_id}: {str(e)}")
            flash(f"An error occurred while setting up your subscription: {str(e)}. Please contact support.", "danger")
        
        return redirect(self.get_redirect())

    @expose('/purchase_credits/<string:package_id>')
    @has_access
    def purchase_credits(self, package_id):
        if not current_user.is_authenticated:
            flash("Please log in to purchase credits.", "warning")
            return redirect(appbuilder.get_url_for_login)

        if not payment_manager.stripe_secret_key:
            flash("Stripe payments are not configured by the administrator. Please contact support.", "danger")
            return redirect(self.get_redirect())

        credit_packages = current_app.config.get('CREDIT_PACKAGES', {})
        package_info = credit_packages.get(package_id)

        if not package_info:
            flash("Invalid credit package selected.", "danger")
            return redirect(self.get_redirect())

        stripe_price_id = package_info.get('stripe_price_id')
        if not stripe_price_id or 'replace_me' in stripe_price_id:
            flash(f"The payment configuration for '{package_info['name']}' is not yet complete. Please contact support.", "warning")
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
                flash("Could not initiate Stripe Checkout for credits. Please try again or contact support.", "danger")
        except Exception as e:
            # In a production app, log this error: current_app.logger.error(f"Stripe credit purchase error for user {current_user.id}, package {package_id}: {str(e)}")
            flash(f"An error occurred while purchasing credits: {str(e)}. Please contact support.", "danger")

        return redirect(self.get_redirect())
    # End of Cascade additions for Stripe Checkout routes

appbuilder.add_view(
    BillingProfileView,
    "Billing Profile",
    icon="fa-credit-card",
    label=("Billing"),
    category="Profile",
    category_icon="fa-user",
)

# Added by Cascade: Stripe Webhook Endpoint
@appbuilder.app.route('/stripe-webhooks', methods=['POST'])
def stripe_webhook():
    # Ensure this environment variable is set on PythonAnywhere
    webhook_secret = current_app.config.get('STRIPE_WEBHOOK_SECRET') 
    if not webhook_secret:
        print("CRITICAL ERROR: Stripe webhook secret (STRIPE_WEBHOOK_SECRET) is not configured in the application environment.")
        # It's crucial Stripe gets a 200 for successfully delivered but unprocessable events if secret is missing on our end,
        # otherwise Stripe will keep retrying. A 500 might be more indicative of server error, but Stripe might retry those too.
        # For now, let's signal we received it but can't process it due to our config issue.
        return jsonify(status="error", message="Webhook secret not configured on server. Event received but not processed."), 200

    payload_string = request.data.decode('utf-8')
    sig_header = request.headers.get('Stripe-Signature')

    if not payload_string or not sig_header:
        print("Webhook Error: Missing payload or signature header.")
        return jsonify(error="Missing payload or signature"), 400

    # The payment_manager instance should be available globally in this file
    success, message = payment_manager.handle_webhook_event(payload_string, sig_header, webhook_secret)

    if success:
        # print(f"Stripe webhook event processed successfully: {message}") # Potentially too verbose for general logs
        return jsonify(status="success", message=message), 200
    else:
        print(f"Stripe webhook processing failed: {message}")
        # Return 400 for client-side errors (e.g., bad signature), 500 for server-side during processing (though handle_webhook_event should catch its own)
        # For now, a general 400 if not successful, as Stripe might retry 500s.
        return jsonify(error=message), 400
# End of Cascade addition for Stripe Webhook

# Added by Cascade: Video Overlay Tool
class VideoOverlayView(BaseView):
    route_base = "/video_overlay"
    default_view = "show"

    @expose('/')
    @has_access
    def show(self):
        # This will render the template from app/templates/video_overlay.html
        return self.render_template('video_overlay.html')

appbuilder.add_view(
    VideoOverlayView,
    "Video Overlay Tool",
    icon="fa-video-camera",
    label="Video Overlay",
    category="Tools",
    category_icon="fa-wrench"
)
# End of Cascade addition


# Added by Cascade: CASI Tool
class CasiView(BaseView):
    route_base = "/casi"
    default_view = "tool"

    @expose('/', methods=['GET', 'POST'])
    @has_access
    def tool(self):
        # Initialize session state if it doesn't exist
        # Initialize session state for conversation and API keys
        if 'casi_thread' not in session:
            session['casi_thread'] = []
        if 'openai_api_key' not in session:
            session['openai_api_key'] = ''
        if 'anthropic_api_key' not in session:
            session['anthropic_api_key'] = ''
        if 'openrouter_api_key' not in session:
            session['openrouter_api_key'] = ''

        # Default prompts and values
        prompts = {
            "generator": "Formalize and expand this idea.",
            "critic": "Analyze and critique this idea."
        }
        
        # Available backends for the dropdowns
        # Note: "groq" is not yet wired; "openrouter" uses the OpenRouter
        # configuration in webCASI and the session-stored openrouter_api_key.
        available_backends = ["openai", "anthropic", "google", "openrouter"]
        
        # Base context for the template
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
            # Default models come from casi.config based on backend
            "generator_model": getattr(casi.config, "openrouter_model", None),
            "critic_model": getattr(casi.config, "openrouter_model", None),
        }

        # Load any existing CASI history so we can expose has_history for
        # enabling the Download Trace button.
        history = session.get('casi_history') or []
        context['has_history'] = bool(history)

        if request.method == 'POST':
            # Update context with form data, providing defaults
            context['selected_gen_backend'] = request.form.get('generator_backend', 'openai')
            context['selected_crit_backend'] = request.form.get('critic_backend', 'openai')
            # Model IDs (optional overrides)
            context['generator_model'] = request.form.get('generator_model') or context.get('generator_model')
            context['critic_model'] = request.form.get('critic_model') or context.get('critic_model')
            context['generator_prompt'] = request.form.get('generator_prompt')
            context['critic_prompt'] = request.form.get('critic_prompt')
            context['generator_input'] = request.form.get('generator_input')
            context['critic_input'] = request.form.get('critic_input', '') # This will be the generator's output
            context['critic_output'] = request.form.get('critic_output', '') # This is feedback for the generator
            context['max_iterations'] = request.form.get('max_iterations', 5)

            action = request.form.get('action')

            # Handle trace download first so it applies regardless of other
            # actions that may also be posted from this form.
            if action == 'download_trace':
                if history:
                    trace_text = casi.format_history_as_text(history)
                    response = current_app.response_class(
                        trace_text,
                        mimetype='text/plain; charset=utf-8',
                    )
                    response.headers['Content-Disposition'] = 'attachment; filename=casi_trace.txt'
                    return response
                else:
                    flash('No CASI history available to download yet.', 'warning')

            # Handle saving API keys to session
            if action == 'save_keys':
                session['openai_api_key'] = request.form.get('openai_api_key', '')
                session['anthropic_api_key'] = request.form.get('anthropic_api_key', '')
                session['openrouter_api_key'] = request.form.get('openrouter_api_key', '')
                flash('API keys have been updated for your session.', 'info')

            elif action == 'run_generator':
                # Determine which API key to use
                api_key = None
                if context['selected_gen_backend'] == 'openai':
                    api_key = session.get('openai_api_key')
                elif context['selected_gen_backend'] == 'anthropic':
                    api_key = session.get('anthropic_api_key')
                elif context['selected_gen_backend'] == 'openrouter':
                    api_key = session.get('openrouter_api_key')

                # Determine model to use (override or default)
                gen_model = context.get('generator_model')
                if not gen_model:
                    gen_model = getattr(casi.config, f"{context['selected_gen_backend']}_model", None)

                gen_output, _ = casi.generator(
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
                # Determine which API key to use
                api_key = None
                if context['selected_crit_backend'] == 'openai':
                    api_key = session.get('openai_api_key')
                elif context['selected_crit_backend'] == 'anthropic':
                    api_key = session.get('anthropic_api_key')
                elif context['selected_crit_backend'] == 'openrouter':
                    api_key = session.get('openrouter_api_key')

                # Determine model to use (override or default)
                crit_model = context.get('critic_model')
                if not crit_model:
                    crit_model = getattr(casi.config, f"{context['selected_crit_backend']}_model", None)

                crit_output, _ = casi.critic(
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
                    if not (1 <= max_iterations <= 20):
                        max_iterations = 5
                except (ValueError, TypeError):
                    max_iterations = 5

                # Determine API keys to use
                if context['selected_gen_backend'] == 'openai':
                    gen_api_key = session.get('openai_api_key')
                elif context['selected_gen_backend'] == 'anthropic':
                    gen_api_key = session.get('anthropic_api_key')
                elif context['selected_gen_backend'] == 'openrouter':
                    gen_api_key = session.get('openrouter_api_key')
                else:
                    gen_api_key = None

                if context['selected_crit_backend'] == 'openai':
                    crit_api_key = session.get('openai_api_key')
                elif context['selected_crit_backend'] == 'anthropic':
                    crit_api_key = session.get('anthropic_api_key')
                elif context['selected_crit_backend'] == 'openrouter':
                    crit_api_key = session.get('openrouter_api_key')
                else:
                    crit_api_key = None

                # Determine models to use (override or default)
                gen_model = context.get('generator_model')
                if not gen_model:
                    gen_model = getattr(casi.config, f"{context['selected_gen_backend']}_model", None)

                crit_model = context.get('critic_model')
                if not crit_model:
                    crit_model = getattr(casi.config, f"{context['selected_crit_backend']}_model", None)

                # Run the full cycle
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

                # Persist CASI history in the session so other views (e.g. dropdown UI)
                # can expose it for trace download.
                session['casi_history'] = results.get('history', [])

                # Also update local history reference so has_history reflects
                # the most recent run in this request.
                history = session['casi_history']
                context['has_history'] = bool(history)

                # Update context with final results and history
                context['generator_output'] = results.get('final_generator_output', '')
                context['critic_output'] = results.get('final_critic_output', '')
                context['cycle_history'] = results.get('history', [])
                context['max_iterations'] = max_iterations # Persist the value in the form
                flash(f'Automatic cycle completed {len(results.get("history", []))} iterations.', 'success')
        
        # This will render app/templates/casi.html
        return self.render_template('casi.html', **context)

appbuilder.add_view(
    CasiView,
    "CASI Tool",
    icon="fa-exchange",
    label="CASI Tool",
    category="Tools",
    category_icon="fa-wrench"
)
# End of CASI Tool Integration


# --- Bespoke Automata Graph UI View ---
class BespokeGraphView(BaseView):
    route_base = "/bespoke_graph"
    default_view = "index"

    @expose("/")
    @has_access
    def index(self):
        return self.render_template("bespoke/bespoke_graph.html")

appbuilder.add_view(
    BespokeGraphView,
    "Bespoke Automata",
    icon="fa-cogs",
    label="Bespoke Automata",
    category="Tools",
    category_icon="fa-wrench"
)

"""
    Application wide 404 error handler
"""
@appbuilder.app.errorhandler(404)
def page_not_found(e):
    return (
        render_template(
            "404.html", base_template=appbuilder.base_template, appbuilder=appbuilder
        ),
        404,
    )


#db.create_all()
