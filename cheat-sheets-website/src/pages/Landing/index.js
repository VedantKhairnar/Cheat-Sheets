import React from 'react';
import { Link } from 'react-router-dom';
import './style.scss';

function Landing() {
    return (
        <div className="Landing">
            <div className="content">
                <div className="content2">
                    <div>All science cheat sheets at one place.</div>
                    <br />
                    <div>Go to <Link to='/topics'><u>Topics</u></Link> Page for cheat sheets.</div>
                </div>
            </div>
        </div>
    )
}

export default Landing
