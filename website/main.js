(function () {
  'use strict';

  // 行動裝置選單
  var toggle = document.querySelector('.nav-toggle');
  var nav = document.getElementById('site-nav');

  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', String(open));
    });

    nav.addEventListener('click', function (event) {
      if (event.target.tagName === 'A') {
        nav.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // 估價表單前端驗證（純靜態網站，不會實際送出資料）
  var form = document.querySelector('.contact-form');
  if (!form) {
    return;
  }

  var status = form.querySelector('.form-status');

  function setError(field, message) {
    field.classList.add('has-error');
    var error = field.querySelector('.field-error');
    if (!error) {
      error = document.createElement('p');
      error.className = 'field-error';
      field.appendChild(error);
    }
    error.textContent = message;
  }

  function clearError(field) {
    field.classList.remove('has-error');
    var error = field.querySelector('.field-error');
    if (error) {
      error.remove();
    }
  }

  form.addEventListener('submit', function (event) {
    event.preventDefault();

    var valid = true;
    var name = form.querySelector('#name');
    var phone = form.querySelector('#phone');
    var message = form.querySelector('#message');

    [name, phone, message].forEach(function (input) {
      clearError(input.closest('.field'));
    });

    if (!name.value.trim()) {
      setError(name.closest('.field'), '請填寫姓名。');
      valid = false;
    }

    if (!/^[0-9+\-() ]{8,15}$/.test(phone.value.trim())) {
      setError(phone.closest('.field'), '請填寫正確的聯絡電話。');
      valid = false;
    }

    if (message.value.trim().length < 5) {
      setError(message.closest('.field'), '請簡短描述問題（至少 5 個字）。');
      valid = false;
    }

    if (!valid) {
      status.textContent = '尚有欄位需要修正，請確認後再送出。';
      return;
    }

    status.textContent =
      '已收到您的需求（示範用途，本頁面尚未串接後端）。請直接來電 0900-000-000 以加快處理。';
    form.reset();
  });
})();
