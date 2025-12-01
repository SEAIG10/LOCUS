import React from "react";
import { NavLink } from "react-router-dom";

const SidebarNav: React.FC = () => {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `sidebar-link${isActive ? " active" : ""}`;

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">LOCUS</div>
      <nav className="sidebar-nav">
        <NavLink to="/fr1" className={linkClass}>
          FR1 · Home & Location
        </NavLink>
        <NavLink to="/fr2" className={linkClass}>
          FR2 · Visual & Audio
        </NavLink>
        <NavLink to="/fr3" className={linkClass}>
          FR3 · GRU Prediction
        </NavLink>
        <NavLink to="/fr4" className={linkClass}>
          FR4 · Fed Learning
        </NavLink>
        <NavLink to="/fr5" className={linkClass}>
          FR5 · Policy Engine
        </NavLink>
      </nav>
    </aside>
  );
};

export default SidebarNav;
