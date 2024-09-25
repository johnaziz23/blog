---
title: "Welcome to My Blog"
---

# John Aziz's Code Blog

Welcome to my blog! Below are my most-recent posts:

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }})
{% endfor %}
