/* WholeBodyPose project page: copy buttons, click-to-load demo. No dependencies. */
(function () {
  'use strict';

  /* ---------- copy buttons ---------- */
  var copyText = function (text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy') ? resolve() : reject(new Error('copy failed')); }
      catch (e) { reject(e); }
      finally { document.body.removeChild(ta); }
    });
  };
  Array.prototype.forEach.call(document.querySelectorAll('[data-copy]'), function (btn) {
    btn.hidden = false;
    var span = btn.querySelector('span');
    var original = span.textContent;
    btn.addEventListener('click', function () {
      var target = document.getElementById(btn.getAttribute('data-copy'));
      if (!target) { return; }
      copyText(target.textContent.replace(/\s+$/, '') + '\n').then(function () {
        span.textContent = 'Copied';
      }, function () {
        span.textContent = 'Press Ctrl+C';
        var range = document.createRange();
        range.selectNodeContents(target);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }).then(function () {
        setTimeout(function () { span.textContent = original; }, 1800);
      });
    });
  });

  /* ---------- demo: load the Hugging Face Space only on request ---------- */
  var loadBtn = document.getElementById('demoLoad');
  var box = document.getElementById('demoBox');
  if (loadBtn && box) {
    loadBtn.hidden = false;
    loadBtn.addEventListener('click', function () {
      var iframe = document.createElement('iframe');
      iframe.src = box.getAttribute('data-src');
      iframe.title = 'WholeBodyPose demo (Hugging Face Space)';
      iframe.loading = 'lazy';
      iframe.allow = 'camera; clipboard-write';
      iframe.referrerPolicy = 'no-referrer';
      box.innerHTML = '';
      box.appendChild(iframe);
      iframe.focus();
    });
  }
})();
