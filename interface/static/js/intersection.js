document.addEventListener('DOMContentLoaded', () => {
    const hiddenSections = document.querySelectorAll('.section-hidden');
    const observer = new IntersectionObserver((entries, obs) => {
      entries.forEach(entry => {
        if(entry.isIntersecting){
          entry.target.classList.add('section-visible');
          obs.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });
  
    hiddenSections.forEach(sec => {
      observer.observe(sec);
    });
  });
  