"""
Style utilities for EIS Analyzer.
"""
import os


def get_custom_css() -> str:
    """Load custom CSS from file."""
    css_path = os.path.join(os.path.dirname(__file__), 'custom.css')
    with open(css_path, 'r') as f:
        return f.read()


def inject_custom_css(st):
    """Inject custom CSS into Streamlit app."""
    css = get_custom_css()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
