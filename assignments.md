---
layout: default
title: Assignments
nav_exclude: false
description: A feed containing all of the weekly assignments
nav_order: 100
---

# Assignments

<!-- Assignments are stored in the `_assignments` directory and rendered according to the layout file, `_layouts/assignment.html`. -->

{% assign assignments = site.assignments | sort: 'ordinal' %}
{% for assignment in assignments %}
{{ assignment }}
{% endfor %}
