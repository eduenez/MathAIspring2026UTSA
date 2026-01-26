---
layout: default
title: Assignments
nav_exclude: false
description: A feed of all assignments
nav_order: 25
---

# Assignments

{% assign assignments = site.assignments | sort: 'ordinal' | reverse %}
{% for assignment in assignments %}
<div class="assignment">
  <h2>{{ assignment.title }}</h2>
  {% if assignment.due_date %}
  <span class="assignment-meta">
    Due: <b>{{ assignment.due_date | date: '%a %e %B' }}</b>
  </span>
  {% endif %}
  <div>
    {{ assignment.content }}
  </div>
</div>
{% endfor %}
