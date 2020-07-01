Module: {{ objname | escape }}
========{{ underline }}

.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:

{% block classes %}
{% if classes %}

Classes
-------

.. autosummary::
   :toctree: _{{ objname }}/_classes
{% for item in classes %}
   {{ item }}{% endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}

Functions
---------

.. autosummary::
   :toctree: _{{ objname }}/_functions

{% for item in functions %}
   {{ item }}{% endfor %}
{% endif %}
{% endblock %}
