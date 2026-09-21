(() => {
  const storageKey = "ironmlx-theme";
  const themes = new Set(["system", "light", "dark"]);

  let selected = "system";
  try {
    const stored = localStorage.getItem(storageKey);
    if (stored && themes.has(stored)) selected = stored;
  } catch (_) {
    // Storage can be unavailable in privacy-restricted contexts.
  }

  document.documentElement.dataset.theme = selected;

  const pickers = () => Array.from(document.querySelectorAll("[data-theme-picker]"));

  const closePicker = (picker, returnFocus = false) => {
    const trigger = picker.querySelector("[data-theme-trigger]");
    const menu = picker.querySelector("[data-theme-menu]");
    trigger.setAttribute("aria-expanded", "false");
    menu.hidden = true;
    if (returnFocus) trigger.focus();
  };

  const syncPicker = (picker) => {
    const trigger = picker.querySelector("[data-theme-trigger]");
    const options = Array.from(picker.querySelectorAll("[data-theme-option]"));
    const active = options.find((option) => option.dataset.themeOption === selected);
    const label = active.querySelector("span:nth-child(2)").textContent;

    trigger.setAttribute("aria-label", `${picker.dataset.themeLabel}: ${label}`);
    trigger.title = label;
    options.forEach((option) => {
      option.setAttribute("aria-checked", String(option === active));
    });
  };

  const apply = (theme, persist = true) => {
    selected = themes.has(theme) ? theme : "system";
    document.documentElement.dataset.theme = selected;
    if (persist) {
      try {
        if (selected === "system") localStorage.removeItem(storageKey);
        else localStorage.setItem(storageKey, selected);
      } catch (_) {
        // Theme switching still works for the current page without storage.
      }
    }
    pickers().forEach(syncPicker);
  };

  const openPicker = (picker, focusTarget = "selected") => {
    pickers().forEach((other) => {
      if (other !== picker) closePicker(other);
    });
    const trigger = picker.querySelector("[data-theme-trigger]");
    const menu = picker.querySelector("[data-theme-menu]");
    const options = Array.from(picker.querySelectorAll("[data-theme-option]"));
    trigger.setAttribute("aria-expanded", "true");
    menu.hidden = false;
    const target = focusTarget === "last"
      ? options.at(-1)
      : options.find((option) => option.dataset.themeOption === selected);
    target.focus();
  };

  document.addEventListener("DOMContentLoaded", () => {
    pickers().forEach((picker) => {
      const trigger = picker.querySelector("[data-theme-trigger]");
      const menu = picker.querySelector("[data-theme-menu]");
      const options = Array.from(picker.querySelectorAll("[data-theme-option]"));

      syncPicker(picker);

      trigger.addEventListener("click", (event) => {
        event.stopPropagation();
        if (menu.hidden) openPicker(picker);
        else closePicker(picker, true);
      });

      trigger.addEventListener("keydown", (event) => {
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          openPicker(picker, event.key === "ArrowUp" ? "last" : "selected");
        }
      });

      options.forEach((option) => {
        option.addEventListener("click", () => {
          apply(option.dataset.themeOption);
          closePicker(picker, true);
        });
      });

      menu.addEventListener("keydown", (event) => {
        const index = options.indexOf(document.activeElement);
        if (event.key === "Escape") {
          event.preventDefault();
          closePicker(picker, true);
        } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          const step = event.key === "ArrowDown" ? 1 : -1;
          options[(index + step + options.length) % options.length].focus();
        } else if (event.key === "Home" || event.key === "End") {
          event.preventDefault();
          options[event.key === "Home" ? 0 : options.length - 1].focus();
        }
      });
    });

    document.addEventListener("click", (event) => {
      pickers().forEach((picker) => {
        if (!picker.contains(event.target)) closePicker(picker);
      });
    });
  });

  window.addEventListener("storage", (event) => {
    if (event.key === storageKey) apply(event.newValue || "system", false);
  });
})();
