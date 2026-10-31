import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SignBridge | Real-Time ISL Multimodal AI Communication Platform',
  description: 'Bridging Indian Sign Language (ISL) users and spoken-language communicators through real-time temporal computer vision, streaming speech AI, and conversational intelligence.'
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#F7F9FC] text-[#14264F] antialiased selection:bg-[#12B8C2] selection:text-white min-h-screen">
        {children}
      </body>
    </html>
  );
}
