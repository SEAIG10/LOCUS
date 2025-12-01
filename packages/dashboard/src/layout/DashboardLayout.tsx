import React from "react";
import SidebarNav from "../components/SidebarNav";
import TopBar from "../components/TopBar";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  return (
    <div className="app-root">
      <SidebarNav />
      <div className="app-main">
        <TopBar />
        <main className="app-content">{children}</main>
      </div>
    </div>
  );
};

export default DashboardLayout;
