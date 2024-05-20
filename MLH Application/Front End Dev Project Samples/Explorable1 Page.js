"use client"

import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';

const Home = () => {
  const [isLocationAllowed, setIsLocationAllowed] = useState(false);
  const [isAudioAllowed, setIsAudioAllowed] = useState(false);

  useEffect(() => {
    // Ask for location access
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          // User granted access to location
          setIsLocationAllowed(true);
        },
        (error) => {
          console.error('Error getting location:', error.message);
          // Handle error or show a message to the user
        }
      );
    } else {
      console.error('Geolocation is not supported by this browser.');
      // Handle the case where geolocation is not supported
    }

    // Ask for microphone access
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(() => {
          // User granted access to audio
          setIsAudioAllowed(true);
        })
        .catch((error) => {
          console.error('Error getting microphone access:', error.message);
          // Handle error or show a message to the user
        });
    } else {
      console.error('getUserMedia is not supported by this browser.');
      // Handle the case where getUserMedia is not supported
    }
  }, []);

  if (!isLocationAllowed || !isAudioAllowed) {
    // You can customize the message or UI for users who haven't granted permissions
    return (
      <div className="text-white text-4xl mt-40 p-30">
        <p>Please grant location and audio access to use this website.</p>
      </div>
    );
  }

  // Render your main content if both location and audio access are granted
  return (
    <section className="text-white text-5xl mt-40">
      <div className="flex">
        <div className="w-2/3 flex flex-col ml-32">
          <p className="text-4xl font-bold text-black">Creating safe Guides </p>
          <p className="text-4xl font-bold text-black mt-5">to your Destinations</p>
          <Link href="/enterdest">
            <button className="w-28 mt-10 p-3 text-sm text-white bg-[#7975FF] rounded-lg">Get Started</button>
          </Link>
        </div>
        <div className="w-1/3 " style={{ position: 'fixed', bottom: 0, right: 0 }}>
          <Image height={250} width={300} src="/assets/phone.png" />
        </div>
      </div>
    </section>
  );
};

export default Home;
