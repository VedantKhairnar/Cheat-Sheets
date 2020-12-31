import React from 'react';
import { Link } from 'react-router-dom';
import './style.scss';

function NotFound() {
    return (
        <div className="NotFound">
            <div>Not Found!</div>
            <div>Go back to <Link to='/'><u>Home</u></Link>.</div>
        </div>
    )
}

export default NotFound
