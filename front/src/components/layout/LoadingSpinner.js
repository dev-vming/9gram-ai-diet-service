window.addEventListener('DOMContentLoaded', () => {
  const dp = new DecayingPreloader('.pl');
});

class DecayingPreloader {
  particles = [];
  totalParticles = 120;
  replayTimeout = null;

  constructor(el) {
    this.el = document.querySelector(el);
    this.particleGroup = this.el?.querySelector('[data-particles]');
    this.worm = this.el?.querySelector('[data-worm]');

    this.init();
  }
  init() {
    this.spawnParticles(this.totalParticles);
    this.worm?.addEventListener(
      'animationend',
      this.replayParticles.bind(this)
    );
  }
  createParticle(x, y, r, delay) {
    const particle = new DecayParticle(x, y, r, delay);
    this.particleGroup?.appendChild(particle.g);
    // animation params
    particle.gAnimation = particle.g.animate(
      [
        { transform: `translate(${particle.x}px,0)` },
        { transform: `translate(${particle.x + particle.dx}px,0)` },
      ],
      { delay: particle.delay, duration: particle.duration, easing: 'linear' }
    );
    particle.cAnimation = particle.c.animate(
      [
        { opacity: 1, transform: `translate(0,${particle.y}px) scale(1)` },
        {
          opacity: 1,
          transform: `translate(0,${particle.y + particle.dy}px) scale(0)`,
        },
      ],
      { delay: particle.delay, duration: particle.duration, easing: 'ease-in' }
    );
    // finally create the particle
    this.particles.push(particle);
  }
  replayParticles() {
    const movingClass = 'pl__worm--moving';
    const timeout = 800;
    // retrigger the worm animation
    this.worm.classList.remove(movingClass);
    clearTimeout(this.replayTimeout);

    this.replayTimeout = setTimeout(() => {
      this.worm.classList.add(movingClass);
      // restart the particles
      this.particles.forEach((particle) => {
        particle.gAnimation.finish();
        particle.gAnimation.play();
        particle.cAnimation.finish();
        particle.cAnimation.play();
      });
    }, timeout);
  }
  spawnParticles(count = 1) {
    const centerXY = 64;
    const radius = 56;
    const loops = 4;
    const maxDelayPerLoop = 2000;
    const particlesPerLoop = Math.round(this.totalParticles / loops);
    const angleOffset = -2;
    const particleRadius = 7;

    for (let c = 0; c < count; ++c) {
      // place along the ring
      const percent = Utils.easeInOutCubic(
        (c % particlesPerLoop) / particlesPerLoop
      );
      const angle = 360 * percent + angleOffset;
      const x = centerXY + radius * Math.sin(Utils.degToRad(angle));
      const y = centerXY - radius * Math.cos(Utils.degToRad(angle));
      const loopsCompleted = Math.floor(c / particlesPerLoop);
      const delay =
        maxDelayPerLoop * percent + maxDelayPerLoop * loopsCompleted;

      this.createParticle(x, y, particleRadius, delay);
    }
  }
}
class DecayParticle {
  duration = 500;
  dx = Utils.randomFloat(-16, 16);
  dy = Utils.randomFloat(32, 64);
  // group
  gAnimation = null;
  // circle
  cAnimation = null;

  constructor(x = 0, y = 0, r = 1, delay = 0) {
    this.x = x;
    this.y = y;
    this.r = r;
    this.delay = delay;
    // namespace
    const ns = 'http://www.w3.org/2000/svg';
    // group to move horizontally in the animation
    const g = document.createElementNS(ns, 'g');
    g.setAttributeNS(null, 'transform', `translate(${x},0)`);
    // circle to move vertically in the animation
    const circle = document.createElementNS(ns, 'circle');
    circle.setAttributeNS(null, 'opacity', '0');
    circle.setAttributeNS(null, 'r', `${this.r}`);
    circle.setAttributeNS(null, 'transform', `translate(0,${y})`);
    circle.setAttributeNS(null, 'fill', 'var(--primary)');

    this.g = g;
    this.c = circle;

    this.g.appendChild(this.c);
  }
}
class Utils {
  static degToRad(deg) {
    return (deg * Math.PI) / 180;
  }
  // ease methods from https://gist.github.com/gre/1650294
  static easeInOutCubic(t) {
    return t < 0.5 ? 4 * t ** 3 : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
  }
  static randomFloat(min = 0, max = 2 ** 32) {
    const percent = crypto.getRandomValues(new Uint32Array(1))[0] / 2 ** 32;
    const relativeValue = (max - min) * percent;
    const plusMin = min + relativeValue;

    return +plusMin.toFixed(3);
  }
}
