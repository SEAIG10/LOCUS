import React from "react";

interface TimelineItemProps {
  time: string;
  label: string;
  highlight?: boolean;
}

const TimelineItem: React.FC<TimelineItemProps> = ({ time, label, highlight }) => {
  return (
    <div className={`timeline-item${highlight ? " timeline-item--highlight" : ""}`}>
      <div className="timeline-time">{time}</div>
      <div className="timeline-label">{label}</div>
    </div>
  );
};

export default TimelineItem;
