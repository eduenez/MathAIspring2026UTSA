---
layout: page
title: Contact me
description: Contact information of all the course staff members.
nav_order: 50
---


<!--
# Staff

Staff information is stored in the `_staffers` directory and rendered according to the layout file, `_layouts/staffer.html`.
-->

# Instructor

{% assign instructors = site.staffers | where: 'role', 'Instructor' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}
