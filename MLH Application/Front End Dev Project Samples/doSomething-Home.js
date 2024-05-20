import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import ActivityCard from '/Users/yasharya/dosomething/my-activity-app/src/ components/ActivityCard.js'; 
import leftIcon from '/Users/yasharya/dosomething/my-activity-app/src/assets/multiply.png';  
import rightIcon from '/Users/yasharya/dosomething/my-activity-app/src/assets/check.png'; 
import locationIcon from '/Users/yasharya/dosomething/my-activity-app/src/assets/placeholder.png'; 
import settingsIcon from '/Users/yasharya/dosomething/my-activity-app/src/assets/settings.png'; 
import { fetchActivities } from '/Users/yasharya/dosomething/my-activity-app/src/Api.js'; 

function Home() {
    const [activities, setActivities] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [userLocation, setUserLocation] = useState('Fetching location...');

    useEffect(() => {
        navigator.geolocation.getCurrentPosition(async (position) => {
            const { latitude, longitude } = position.coords;
            const activitiesFromAPI = await fetchActivities(latitude, longitude);
            setActivities(activitiesFromAPI);
            setCurrentIndex(0); 
        }, () => {
            setUserLocation('Location access denied');
        });

        fetch('https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=latitude&longitude=longitude&localityLanguage=en')
            .then(response => response.json())
            .then(data => setUserLocation(data.locality || data.city || 'Unknown location'))
            .catch(() => setUserLocation('Location unavailable'));
    }, []);

    const handleSwipe = (direction) => {
        console.log(direction, ':', activities[currentIndex]?.name);
        const nextIndex = currentIndex + (direction === 'Liked' ? 1 : -1);
        if (nextIndex < activities.length && nextIndex >= 0) {
            setCurrentIndex(nextIndex);
        } else {
            console.log('Reached the end or beginning of the deck');
        }
    };
    

    return (
        <div style={{ position: 'relative', width: '320px', margin: 'auto' }}>
            <div style={{ position: 'absolute', top: '10px', right: '10px' }}>
                <Link to="/preferences">
                    <img src={settingsIcon} alt="Settings" style={{ width: '30px', height: '30px' }} />
                </Link>
            </div>
            <h1>doSmth.com</h1>
            {activities.length > 0 && currentIndex < activities.length ? (
                <div>
                <button onClick={() => handleSwipe('Disliked')} style={{ position: 'absolute', bottom: '10px', left: '10px', zIndex: 1, background: 'none', border: 'none', cursor: 'pointer', width: '70px', height: '70px' }}>
                    <img src={leftIcon} alt="Dislike" style={{ width: '100%', height: '100%' }} />
                </button>
                <ActivityCard
                    activity={activities[currentIndex]}
                    onSwipeLeft={() => handleSwipe('Disliked')}
                    onSwipeRight={() => handleSwipe('Liked')}
                />
                <button onClick={() => handleSwipe('Liked')} style={{ position: 'absolute', bottom: '10px', right: '10px', zIndex: 1, background: 'none', border: 'none', cursor: 'pointer', width: '70px', height: '70px' }}>
                    <img src={rightIcon} alt="Like" style={{ width: '100%', height: '100%' }} />
                </button>
            </div>
            ) : (
                <p>No more activities!</p>
            )}
            <div style={{ position: 'absolute', bottom: '-350px', width: '100%', textAlign: 'center', padding: '10px 0', fontSize: '16px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <img src={locationIcon} alt="Location" style={{ width: '24px', height: '24px', marginRight: '5px' }} />
                <span>{userLocation}</span>
            </div>
        </div>
    );
    
}

export default Home;