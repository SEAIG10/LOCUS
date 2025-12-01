import React from "react";

const TopBar: React.FC = () => {
  return (
    <header className="topbar">
      <div className="topbar-title">LOCUS Dashboard</div>
      <div className="topbar-right">
        <span className="topbar-badge">Demo</span>
      </div>
    </header>
  );
};

export default TopBar;
