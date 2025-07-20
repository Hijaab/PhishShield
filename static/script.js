document.addEventListener('DOMContentLoaded', () => {
  const themeToggle = document.getElementById('theme-toggle');
  const body = document.body;
  const dropArea = document.getElementById('drop-area');
  const imageUpload = document.getElementById('image-upload');
  const form = document.getElementById('upload-form');
  const loader = document.getElementById('loading');
  const preview = document.getElementById('preview');

  // --- Theme Persistence ---
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'dark') {
    body.classList.add('dark');
    themeToggle.checked = true;
  }

  themeToggle.addEventListener('change', () => {
    const isDark = themeToggle.checked;
    body.classList.toggle('light', !isDark);
    body.classList.toggle('dark', isDark);
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  });

  // --- Image Preview Before Submit ---
  imageUpload.addEventListener('change', () => {
    const file = imageUpload.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = e => {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
      };
      reader.readAsDataURL(file);
    }
  });

  // --- Drag and Drop Support ---
  ['dragenter', 'dragover'].forEach(event => {
    dropArea.addEventListener(event, e => {
      e.preventDefault();
      e.stopPropagation();
      dropArea.classList.add('highlight');
    });
  });

  ['dragleave', 'drop'].forEach(event => {
    dropArea.addEventListener(event, e => {
      e.preventDefault();
      e.stopPropagation();
      dropArea.classList.remove('highlight');
    });
  });

  dropArea.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      imageUpload.files = files;

      // Show preview before auto-submit
      const reader = new FileReader();
      reader.onload = e => {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
      };
      reader.readAsDataURL(files[0]);

      form.submit();
    }
  });

  // --- Show loader on submit ---
  form.addEventListener('submit', () => {
    loader.classList.remove('hidden');
  });

  // --- Render Chart ---
  if (window.modelScores && document.getElementById('chart')) {
    const ctx = document.getElementById('chart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: Object.keys(modelScores),
        datasets: [{
          label: 'Model Score (%)',
          data: Object.values(modelScores),
          backgroundColor: '#3b82f6',
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 100
          }
        }
      }
    });
  }
});
// --- Render Chart ---
if (window.modelScores && document.getElementById('chart')) {
  const ctx = document.getElementById('chart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Object.keys(modelScores),
      datasets: [{
        label: 'Model Score (%)',
        data: Object.values(modelScores),
        backgroundColor: '#3b82f6',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100
        }
      }
    }
  });
}
