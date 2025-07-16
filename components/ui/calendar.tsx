"use client";

import * as React from "react";
import Datepicker from "react-tailwindcss-datepicker";

import { cn } from "@/lib/utils";
import { buttonVariants } from "@/components/ui/button";

export type CalendarProps = {
  value: { startDate: string | null; endDate: string | null };
  onChange: (value: {
    startDate: string | null;
    endDate: string | null;
  }) => void;
  className?: string;
};

export function Calendar({ value, onChange, className }: CalendarProps) {
  return (
    <div className={className}>
      <Datepicker value={value} onChange={onChange} showShortcuts={true} />
    </div>
  );
}
Calendar.displayName = "Calendar";

export { Calendar };
