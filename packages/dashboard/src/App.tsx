import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import DashboardLayout from "./layout/DashboardLayout";
import FR1HomeStructurePage from "./pages/FR1HomeStructurePage";
import FR2ContextAwarenessPage from "./pages/FR2ContextAwarenessPage";
import FR3SequentialGRUPage from "./pages/FR3SequentialGRUPage";
import FR4FederatedLearningPage from "./pages/FR4FederatedLearningPage";
import FR5PolicyEnginePage from "./pages/FR5PolicyEnginePage";

const App: React.FC = () => {
  return (
    <DashboardLayout>
      <Routes>
        <Route path="/" element={<Navigate to="/fr1" replace />} />
        <Route path="/fr1" element={<FR1HomeStructurePage />} />
        <Route path="/fr2" element={<FR2ContextAwarenessPage />} />
        <Route path="/fr3" element={<FR3SequentialGRUPage />} />
        <Route path="/fr4" element={<FR4FederatedLearningPage />} />
        <Route path="/fr5" element={<FR5PolicyEnginePage />} />
      </Routes>
    </DashboardLayout>
  );
};

export default App;
