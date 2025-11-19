from flask import render_template, redirect, url_for, flash, request, abort, jsonify, current_app, session
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi, BaseView, expose, has_access
from flask_appbuilder import SimpleFormView
from . import appbuilder, db

from .models import Individual, ListGroup, Subscription, SUBSCRIPTION_TIERS # Ensure Subscription and SUBSCRIPTION_TIERS are imported
from .forms import PE_Form, waitForm
from collections import deque
import openai
import json
import os # Ensure os is imported if not already

# Ensure payment_manager is initialized after db and appbuilder
from .payment_manager import PaymentManager
from .runpod_manager import RunPodManager
from . import webCASI as casi
from .models import UserAPIKey
from flask_login import current_user

# ... (rest of the original code up to CasiView) ...

class CasiView(BaseView):
    route_base = "/casi"
    default_view = "index"

    @expose("/", methods=["GET", "POST"])
    @has_access
    def index(self):
        context = {}

        # Query API keys for dropdowns
        openai_keys = UserAPIKey.query.filter_by(user_id=current_user.id, service_name='openai').all()
        anthropic_keys = UserAPIKey.query.filter_by(user_id=current_user.id, service_name='anthropic').all()
        context['openai_keys'] = openai_keys
        context['anthropic_keys'] = anthropic_keys

        # Load CASI history (if any) from the session and expose has_history
        history = session.get('casi_history') or []
        context['has_history'] = bool(history)

        if request.method == 'POST':
            action = request.form.get('action')

            if action == 'download_trace':
                if history:
                    # Use the helper on the CASI backend to format a text trace
                    trace_text = casi.format_history_as_text(history)
                    response = current_app.response_class(
                        trace_text,
                        mimetype='text/plain; charset=utf-8',
                    )
                    response.headers['Content-Disposition'] = 'attachment; filename=casi_trace.txt'
                    return response
                else:
                    flash('No CASI history available to download yet.', 'warning')

            # Other actions (e.g., running generator/critic/loop) can be handled here
            # without affecting the download behavior.

        return self.render_template('casi_dropdown.html', **context)

appbuilder.add_view(
    CasiView,
    "CASI Tool (Dropdown)",
    icon="fa-exchange",
    label="CASI Tool (Dropdown)",
    category="Tools",
    category_icon="fa-wrench"
)
# End of CASI Tool Integration

# ... (rest of the original code) ...

