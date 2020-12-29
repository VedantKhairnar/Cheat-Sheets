import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './style.scss';

function Header() {

    const [selectedNavLink, setSelectedNavLink] = useState("home");

    function handleNavLinkChange(name) {
        setSelectedNavLink(name);
    }

    return (
        <header className="Header">
            <Link to='/' className="logo" onClick={() => handleNavLinkChange('home')}>
                Cheat Sheets
            </Link>
            <nav className="nav">
                <Link to='/topics' className={`nav-links ${selectedNavLink === 'topics' ? 'active' : ''}`}
                    onClick={() => handleNavLinkChange('topics')}
                >
                    Topics
                </Link>
                <Link to='/aboutus' className={`nav-links ${selectedNavLink === 'aboutus' ? 'active' : ''}`}
                    onClick={() => handleNavLinkChange('aboutus')}
                >
                    About Us
                </Link>
                <a href="https://github.com/VedantKhairnar/Cheat-Sheets" className="nav-links" target="_blank" rel="noreferrer">
                    GitHub Repo
                </a>
            </nav>
        </header >
    )
}

export default Header
