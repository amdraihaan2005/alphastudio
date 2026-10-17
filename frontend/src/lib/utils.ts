import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Builds clean, human-readable citation label from filename and page number.
 */
export function buildDisplayLabel(filename: string, pageNum: number): string {
  const clean = filename.replace(/\.pdf$/i, '').trim();
  const yearMatch = clean.match(/(?:FY|20)(\d{2})/i) || clean.match(/\b(20\d{2})\b/);
  let yearStr = '';
  if (yearMatch) {
    const yr = yearMatch[1];
    yearStr = yr.length === 4 ? `'${yr.slice(2)}` : `'${yr}`;
  }
  
  let company = clean
    .replace(/[_-]+/g, ' ')
    .replace(/\b(FY\d{2,4}|20\d{2}|annual|report|filing|10k|10-k|quarterly|results)\b/gi, '')
    .trim();

  if (!company) {
    company = clean.replace(/[_-]+/g, ' ');
  }

  if (company === company.toUpperCase() && company.length > 4) {
    company = company.charAt(0) + company.slice(1).toLowerCase();
  } else if (company === company.toLowerCase()) {
    company = company.charAt(0).toUpperCase() + company.slice(1);
  }

  const parts = [company];
  if (yearStr && !company.includes(yearStr)) {
    parts.push(yearStr);
  }
  return `${parts.join(' ')}, p. ${pageNum}`;
}
