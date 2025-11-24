---
title: "Robot Learning"
permalink: /robot-learning/
layout: archive
author_profile: true
sidebar:
  nav: "robot_learning_nav"
---

**Robotics** & **Machine Learning** 

<!-- =========================================
     第一板块：Robotics (自动抓取路径包含 /robotics/ 的文件)
     ========================================= -->
<div class="archive-section">
  <h2 id="robotics" class="archive__subtitle" style="color: #2c3e50;">
    🤖 Robotics (机器人学)
  </h2>
  
  {% assign robo_docs = site.robot_learning | where_exp: "item", "item.path contains '/robotics/'" %}
  
  {% if robo_docs.size > 0 %}
    <div class="grid__wrapper">
      {% for post in robo_docs %}
        <div class="archive__item">
          <h3 class="archive__item-title">
            <a href="{{ post.url }}">{{ post.title }}</a>
          </h3>
          <p class="archive__item-excerpt">{{ post.excerpt | strip_html | truncate: 80 }}</p>
        </div>
      {% endfor %}
    </div>
  {% else %}
    <p>🚧 机器人学板块建设中...</p>
  {% endif %}
</div>

<hr style="margin: 30px 0;">

<!-- =========================================
     第二板块：Machine Learning (自动抓取路径包含 /machine_learning/ 的文件)
     ========================================= -->
<div class="archive-section">
  <h2 id="machine-learning" class="archive__subtitle" style="color: #e67e22;">
    🧠 Machine Learning (机器学习)
  </h2>
  
  {% assign ml_docs = site.robot_learning | where_exp: "item", "item.path contains '/machine_learning/'" %}
  
  {% if ml_docs.size > 0 %}
    <div class="grid__wrapper">
      {% for post in ml_docs %}
        <div class="archive__item">
          <h3 class="archive__item-title">
            <a href="{{ post.url }}">{{ post.title }}</a>
          </h3>
          <p class="archive__item-excerpt">{{ post.excerpt | strip_html | truncate: 80 }}</p>
        </div>
      {% endfor %}
    </div>
  {% else %}
    <p>🚧 机器学习板块建设中...</p>
  {% endif %}
</div>