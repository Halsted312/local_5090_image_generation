import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import PrankCreateApp from "./PrankCreateApp";
import PrankPlayApp from "./PrankPlayApp";
import "./index.css";

const path = window.location.pathname;

let ui: React.ReactElement;

if (path.startsWith("/p/")) {
  const [, , slug] = path.split("/");
  ui = <PrankPlayApp slug={slug ?? ""} />;
} else if (path === "/create") {
  ui = <PrankCreateApp />;
} else if (path === "/create/admin") {
  ui = <PrankCreateApp initialSlug="imagine" requirePassword adminPassword="halsted" />;
} else if (path.startsWith("/create/")) {
  const [, , slug] = path.split("/");
  ui = <PrankCreateApp key={slug} initialSlug={slug ?? ""} />;
} else {
  ui = <App />;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {ui}
  </React.StrictMode>,
);
