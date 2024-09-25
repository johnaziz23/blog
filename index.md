---
title: "Welcome to My Blog"
---

# My Blog

Welcome to my blog! Here are my posts:

{% for post in site.posts %}
  ## [{{ post.title }}]({{ post.url }})  
  **Published on:** {{ post.date | date: "%B %d, %Y" }}

  {{ post.excerpt }}
{% endfor %}
