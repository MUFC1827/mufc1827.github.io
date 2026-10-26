const paths = {
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  external: '<path d="M7 17 17 7M7 7h10v10"/>',
  search: '<circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.5 4.5"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
  moon: '<path d="M20 14a8.5 8.5 0 0 1-10-10A8.5 8.5 0 1 0 20 14Z"/>',
  book: '<path d="M12 5c-3-2-6-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1Zm0 0v15"/>',
  code: '<path d="m8 5-7 7 7 7m8-14 7 7-7 7m-3-16-2 18"/>',
  spark: '<path d="m12 3 2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6Z"/>',
  github: '<path d="M9 19c-4.3 1.3-4.3-2-6-2.5m12 5v-3.4c0-1 .1-1.5-.5-2.1 3-.3 6.1-1.5 6.1-6.9 0-1.5-.5-2.7-1.5-3.7.2-.3.7-1.8-.2-3.6 0 0-1.2-.4-3.8 1.4a13 13 0 0 0-6.8 0C5.7 1.4 4.5 1.8 4.5 1.8c-.9 1.8-.4 3.3-.2 3.6-1 1-1.5 2.2-1.5 3.7 0 5.4 3.1 6.6 6.1 6.9-.5.5-.7 1.1-.7 2.1v3.4"/>',
  bookmark: '<path d="M6 3h12v18l-6-4-6 4Z"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  shuffle: '<path d="m17 3 4 4-4 4m0 2 4 4-4 4M3 6h3c5 0 7 12 12 12h3M3 18h3c2 0 3-2 4-4m4-4c1-2 2-4 4-4h3"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  back: '<path d="M20 12H4m6 6-6-6 6-6"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
};
export const icon = (name, size = 20) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.arrow}</svg>`;
export const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
