---
title: "Welcome to John Aziz's Code Blog"
---

# My Blog

Welcome to John Aziz's code blog! Here are my posts:


{% for post in site.posts %}
## {{ post.title }}
Published on: {{ post.date | date: "%B %d, %Y" }}

  {{ post.content | markdownify }}  <!-- This line will render the full post content -->
{% endfor %}

